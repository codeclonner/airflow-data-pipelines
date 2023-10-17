import json
from datetime import timedelta

import airflow
from airflow import DAG
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow.operators.python import PythonVirtualenvOperator
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerTrainingOperator
from airflow.providers.amazon.aws.operators.sagemaker import SageMakerTransformOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule

S3_BUCKET_NAME = 'YOUR_S3_BUCKET_NAME'
REGION_NAME = 'us-east-1'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': airflow.utils.dates.days_ago(1),
    'retries': 0,
    'retry_delay': timedelta(minutes=2),
    'provide_context': True
}

def preprocess(bucket_name):

    import pandas as pd
    import numpy as np
    import boto3
    import os

    my_region = boto3.session.Session().region_name # set the region of the instance

    # set an output path where the trained model will be saved.
    prefix = 'xgboost'
    output_path ='s3://{}/{}/output'.format(bucket_name, prefix)

    # Download file from S3 bucket and load in dataframe (model_data)
    prefix_1 = 'raw'   # Enter your folder where you will upload your dataet file
    data_file = 'train_1.csv'    # Enter the name of your dataset file
    data_location = 's3://{}/{}/{}'.format(bucket_name,prefix_1,data_file)

    df = pd.read_csv(data_location)

    # Check for missing data
    #df.isnull().sum()

    # Here we can see that Coloumn-2 "Pickup_datetime" is an object ---> which we need to convert to "datetime_object" to use in ML algorithms.
    # Pandas can do that easily.
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format= '%Y-%m-%d %H:%M:%S UTC')


    # Since we see from above table Lat & Long are not in correct range of NYC -- we will drop the fields which fall outisde these ranges.
    # we can filter the data using the df.loc function in Pandas.
    df = df.loc[df['pickup_latitude'].between(40,42)]
    df = df.loc[df['pickup_longitude'].between(-75,-72)]
    df = df.loc[df['dropoff_latitude'].between(40,42)]
    df = df.loc[df['dropoff_longitude'].between(-75,-72)]


    # Now let's try to fix the "fare amount" and "passenger-count"
    df = df.loc[df['fare_amount'] > 2.5]   # US$ 2.50 is a minimum fare taxi will charge - so we are considering only those fields who are above $2.50.
    df = df.loc[df['passenger_count'] > 0]


    # Here we can see 1 outlier -- which is 9 passengers, which seem incorrect.
    # Let's drop those outliers
    df = df.loc[df['passenger_count'] <=6]

    # Let's create new columns 'Year', 'month', 'Day' etc... from a single column "pickup_datetime".
    df['year']=df.pickup_datetime.dt.year
    df['month']=df.pickup_datetime.dt.month
    df['day']=df.pickup_datetime.dt.day
    df['weekday']=df.pickup_datetime.dt.weekday
    df['hour']=df.pickup_datetime.dt.hour


    # Let's calculate - distance now.
    def haversine_np(lon1, lat1, lon2, lat2):

        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)

        All args must be of equal length.

        """
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

        c = 2 * np.arcsin(np.sqrt(a))
        km = 6367 * c
        return km


    # Now let's create one more column 'distance'.
    df['distance'] = haversine_np(df['pickup_longitude'],df['pickup_latitude'],df['dropoff_longitude'],df['dropoff_latitude'])


    # We can see above that there are some points "min" -- which has zero distance - let's try to dop those fields.
    df = df.loc[df['distance'] > 0]

    # But before we pass our dataset to a algorithm to create a model --- let's drop the features which we don't need.
    # For e.g. 'key' and 'pickup_datetime' -- becuase we have already extracted all those data in other columns.

    del df['pickup_datetime']


    # Train, Test Split
    train_data, validation_data, test_data = np.split(df.sample(frac=1, random_state=1729), [int(.6*len(df)), int(.8*len(df))])
    print(train_data.shape, validation_data.shape, test_data.shape)


    # Let's create a csv file from this 'train_data' and upload to S3 bucket --- under 'xgboost' prefix.
    train_data.to_csv('train.csv', index=False, header=False)
    boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')

    # Let's perform same steps for validatation data.
    validation_data.to_csv('validate.csv', index=False, header=False)
    boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'validate/validate.csv')).upload_file('validate.csv')


    del test_data['fare_amount']

    test_data.to_csv('test.csv', index=False, header=False)
    boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'test/test.csv')).upload_file('test.csv')

# -----------
# Setup Train and Transform(prediction) task configuration
# -----------

# Seeding sagemaker train and transform config and writing to S3
s3 = S3Hook()
sagemaker_config = [{},{}]
sagemaker_config_json = json.dumps(sagemaker_config)

if not s3.check_for_key(key='task_storage/sagemaker_config.json', bucket_name=S3_BUCKET_NAME):
    s3.load_string(
        string_data=sagemaker_config_json,
        key='task_storage/sagemaker_config.json',
        bucket_name=S3_BUCKET_NAME,
        replace=True
    )

#-----------
### Setup Train and Transofrm(prediction) task configuration
#-----------

def getTrainTransformConfig(**kwargs):

    import boto3
    import json
    import sagemaker
    import datetime
    from sagemaker.amazon.amazon_estimator import get_image_uri
    from sagemaker.estimator import Estimator
    from sagemaker.workflow.airflow import training_config, transform_config_from_estimator


    def serialize_datetime(obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        raise TypeError("Type not serializable")

    bucket_name = kwargs['bucket_name']
    region_name = kwargs['region_name']
    config = {}

    config["job_level"] = {
        "region_name": region_name,
        "run_hyperparameter_opt": "no"
    }

    config["train_model"] = {
        "sagemaker_role": "AirflowSageMakerExecutionRole",
        "estimator_config": {
            "train_instance_count": 1,
            "train_instance_type": "ml.m5.2xlarge",
            "train_volume_size": 5,   # %GB storage
            "train_max_run": 3600,
            "output_path": "s3://{}/xgboost/output".format(bucket_name),
            "hyperparameters": {
                "feature_dim": "178729",
                "epochs": "10",
                "mini_batch_size": "200",
                "num_factors": "64",
                "predictor_type": "regressor",
                "max_depth": "5",
                "eta": "0.2",
                "objective": "reg\:linear",
                "early_stopping_rounds": "10",
                "num_round": "150"
            }
        },
        "inputs": {
            "train": "s3://{}/xgboost/train/train.csv".format(bucket_name),
            "validation": "s3://{}/xgboost/validate/validate.csv".format(bucket_name)
        }
    }

    config["batch_transform"] = {
        "transform_config": {
            "instance_count": 1,
            "instance_type": "ml.c4.xlarge",
            "data": "s3://{}/xgboost/test/".format(bucket_name),
            "data_type": "S3Prefix",
            "content_type": "text/csv",
            "strategy": "SingleRecord",
            "split_type": "Line",
            "output_path": "s3://{}/transform/".format(bucket_name)
        }
    }

    region = config["job_level"]["region_name"]

    boto3_session = boto3.session.Session(region_name=region)
    sagemaker_session = sagemaker.session.Session(boto_session=boto3_session)

    iam = boto3.client('iam', region_name=region)
    role = iam.get_role(RoleName=config["train_model"]["sagemaker_role"])['Role']['Arn']

    container = sagemaker.image_uris.retrieve(region=region_name, framework='xgboost', version='1.7-1')

    train_input = config["train_model"]["inputs"]["train"]
    csv_train_input = sagemaker.inputs.TrainingInput(train_input, content_type='csv')

    validation_input = config["train_model"]["inputs"]["validation"]
    csv_validation_input = sagemaker.inputs.TrainingInput(validation_input, content_type='csv')

    training_inputs = {"train": csv_train_input, "validation": csv_validation_input}
    output_path = config["train_model"]["estimator_config"]["output_path"]

    fm_estimator = Estimator(image_uri=container,
                             role=role,
                             instance_count=1,
                             instance_type='ml.m5.2xlarge',
                             volume_size=5,
                             output_path=output_path,
                             sagemaker_session=sagemaker_session
                             )

    fm_estimator.set_hyperparameters(max_depth=5,
                                     eta=0.2,
                                     objective='reg:linear',
                                     early_stopping_rounds=10,
                                     num_round=150)

    train_config = training_config(estimator=fm_estimator, inputs=training_inputs)

    transform_config = transform_config_from_estimator(
        estimator=fm_estimator,
        task_id="model_tuning" if False else "model_training",
        task_type="tuning" if False else "training",
        **config["batch_transform"]["transform_config"]
    )

    # Write the configuration to S3
    s3 = boto3.resource('s3')
    s3object = s3.Object(bucket_name, 'task_storage/sagemaker_config.json')

    s3object.put(
        Body=(bytes(json.dumps([train_config, transform_config], default=serialize_datetime).encode('UTF-8')))
    )


dag = DAG('ml_pipeline',
          default_args=default_args,
          dagrun_timeout=timedelta(hours=2),
          schedule_interval=None)

s3_sensor = S3KeySensor(task_id='s3_sensor',
                        bucket_name=S3_BUCKET_NAME, bucket_key='raw/train_1.csv', dag=dag)

# Create python operator to call our preprocess function (preprocess.py file).
preprocess_task = PythonVirtualenvOperator(task_id='preprocess',
                                           python_callable=preprocess,
                                           op_kwargs={'bucket_name': S3_BUCKET_NAME},
                                           requirements=["boto3", "pandas","numpy","fsspec","s3fs"],
                                           use_dill=True,
                                           system_site_packages=False,
                                           dag=dag)

# Generate and fetch train and transform config
get_traintransform_config_task  = PythonVirtualenvOperator(
    task_id='generate_config',
    python_callable=getTrainTransformConfig,
    provide_context=True, # the provide_context=True parameter specifies that the Python callable (or function) should receive additional keyword arguments that provide context about the current execution.
    op_kwargs={'bucket_name': S3_BUCKET_NAME, 'region_name': REGION_NAME},
    requirements=["sagemaker","boto3"],
    use_dill=True,  # Use dill to serialize the Python callable
    system_site_packages=False,  # Do not include system site packages in the virtual environment
    dag=dag
)

# launch sagemaker training job and wait until it completes
train_model_task = SageMakerTrainingOperator(
    task_id='model_training',
    dag=dag,
    config=json.loads(s3.read_key(bucket_name=S3_BUCKET_NAME, key='task_storage/sagemaker_config.json'))[0],
    aws_conn_id='aws_default',
    wait_for_completion=True,
    check_interval=30)

# launch sagemaker batch transform job and wait until it completes
batch_transform_task = SageMakerTransformOperator(
    task_id='predicting',
    dag=dag,
    config=json.loads(s3.read_key(bucket_name=S3_BUCKET_NAME, key='task_storage/sagemaker_config.json'))[1],
    aws_conn_id='aws_default',
    wait_for_completion=True,
    check_interval=30,
    trigger_rule=TriggerRule.ONE_SUCCESS)

# set the dependencies between tasks
s3_sensor >> preprocess_task >> get_traintransform_config_task >> train_model_task >> batch_transform_task