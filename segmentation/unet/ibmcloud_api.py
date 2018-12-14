import ibm_boto3
from ibm_botocore.client import Config
from watson_machine_learning_client import WatsonMachineLearningAPIClient

import json, platform, sys
from uuid import uuid4
from urllib.request import urlopen

class IBMCloudAPI(object):
    def __init__(self, cos_credentials, cos_service_endpoint, wml_credentials):
        self.cos_credentials = cos_credentials
        self.cos_service_endpoint = cos_service_endpoint
        self.wml_credentials = wml_credentials
        self.cos = ibm_boto3.resource('s3',
                             ibm_api_key_id=cos_credentials['apikey'],
                             ibm_service_instance_id=cos_credentials['resource_instance_id'],
                             ibm_auth_endpoint='https://iam.bluemix.net/oidc/token',
                             config=Config(signature_version='oauth'),
                             endpoint_url=cos_service_endpoint)
        
        self.client = WatsonMachineLearningAPIClient(wml_credentials)

    def create_cos_bucket(self, bucket_name):
        if not self.cos.Bucket(bucket_name) in self.cos.buckets.all():
            print('Creating bucket "{}"... '.format(bucket_name), end="")
            try:
                self.cos.create_bucket(Bucket=bucket_name,
                                  CreateBucketConfiguration={'LocationConstraint': 'us-standard'},
                                 )
                print("done")
            except ibm_boto3.exceptions.ibm_botocore.client.ClientError as e:
                    print('Error: {}.'.format(e.response['Error']['Message']))

    def empty_cos_bucket(self, bucket_name):
        for obj in self.get_bucket_items(bucket_name):
            obj.delete()
                
    def delete_cos_bucket(self, bucket_name, force=False):
        if force:
            self.empty_cos_bucket(bucket_name)
            self.cos.Bucket(bucket_name).delete()
        else:
            try:
                self.cos.Bucket(bucket_name).delete()
            except ibm_boto3.exceptions.ibm_botocore.client.ClientError as e:
                print('Error: {}.'.format(e.response['Error']['Message']))
                print("Set 'force=True' to delete bucket and it's contents")
                
    def upload_to_bucket(self, bucket_name, file_path=None, link=None):
        bucket_obj = self.cos.Bucket(bucket_name)

        if link is not None:
            if platform.system() == "Darwin" or "Linux": #Darwin = Mac
                filename=link.split('/')[-1]
            elif platform.system() == "Windows":
                filename=link.split('\\')[-1]

            print('Uploading data "{}" to bucket "{}"...'.format(filename, bucket_name), end="")
            with urlopen(link) as data:
                bucket_obj.upload_fileobj(data, filename)
                print(' done')

        elif file_path is not None:
            if platform.system() == "Darwin" or "Linux": #Darwin = Mac
                filename=file_path.split('/')[-1]
            elif platform.system() == "Windows":
                filename=file_path.split('\\')[-1]

            print('Uploading data "{}" to bucket "{}"...'.format(filename, bucket_name), end="")
            bucket_obj.upload_file(file_path, filename)
            print(" done")

    def get_cos_buckets(self):
        return list(self.cos.buckets.all())

    def get_bucket_items(self, bucket_name):
        return list(self.cos.Bucket(bucket_name).objects.all())

    def create_model_definition(self, filename, metadata):
        definition_details = self.client.repository.store_definition(filename, metadata)
        definition_uid = self.client.repository.get_definition_uid(definition_details)
        return definition_uid
    
    def list_repository(self):
        return self.client.repository.list()
    
    def list_model_definitions(self):
        return self.client.repository.list_definitions()
    
    def list_training_definitions(self):
        return self.client.training.list()
    
    def delete_model_definition(self, definition_uid):
        self.client.repository.delete(definition_uid)
    
    def delete_training_definition(self, run_uid):
        self.client.training.delete(run_uid)

    def training_run(self, definition_uid, training_configuration_metadata, asynchronous=True, log=False):
        training_run_details = self.client.training.run(definition_uid, training_configuration_metadata, asynchronous=asynchronous)
        training_run_guid = self.client.training.get_run_uid(training_run_details)
        if log:
            self.client.training.monitor_logs(training_run_guid)
        return training_run_guid, training_run_details
    
    def cancel_training_run(self, run_uid):
        self.client.training.cancel(run_uid)
            
    def train_in_cloud(self, model_file, buckets, def_name, asynchronous=True, log=False, command=[]):
        # model definition
        model_metadata = {
            self.client.repository.DefinitionMetaNames.NAME              : "(Definition) " + def_name,
            #self.client.repository.DefinitionMetaNames.AUTHOR_EMAIL      : "christoffer.hjort@ibm.com",
            self.client.repository.DefinitionMetaNames.FRAMEWORK_NAME    : "tensorflow",
            self.client.repository.DefinitionMetaNames.FRAMEWORK_VERSION : "1.5",
            self.client.repository.DefinitionMetaNames.RUNTIME_NAME      : "python",
            self.client.repository.DefinitionMetaNames.RUNTIME_VERSION   : "3.5",
            self.client.repository.DefinitionMetaNames.EXECUTION_COMMAND : " ".join(command)
        }
        definition_uid = self.create_model_definition(model_file, model_metadata)
        print( "definition_uid: ", definition_uid )

        # training definition
        training_metadata = {
        self.client.training.ConfigurationMetaNames.NAME         : "(Training) " + def_name,
        #self.client.training.ConfigurationMetaNames.AUTHOR_EMAIL : "christoffer.hjort@ibm.com",
        self.client.training.ConfigurationMetaNames.COMPUTE_CONFIGURATION : {
            "name" : "k80"
        },
        self.client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCE : {
           "connection" : { 
              "endpoint_url"          : buckets["data_bucket"]["endpoint"],
              "aws_access_key_id"     : self.cos_credentials["cos_hmac_keys"]["access_key_id"],
              "aws_secret_access_key" : self.cos_credentials["cos_hmac_keys"]["secret_access_key"]
              },
           "source" : {
              "bucket" : buckets["data_bucket"]["name"],
              },
              "type" : "s3"
           },
        self.client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE: {
           "connection" : {
              "endpoint_url"          : buckets["result_bucket"]["endpoint"],
              "aws_access_key_id"     : self.cos_credentials["cos_hmac_keys"]["access_key_id"],
              "aws_secret_access_key" : self.cos_credentials["cos_hmac_keys"]["secret_access_key"]
              },
              "target" : {
                 "bucket" : buckets["result_bucket"]["name"],
              },
              "type" : "s3"
           }
        }
        
        run_uid, run_details = self.training_run(definition_uid, training_metadata, asynchronous=asynchronous, log=log)
        print("Run uid: ", run_uid)
        print("Run details:")
        print(json.dumps(run_details, indent=1))
        return definition_uid, run_uid
        
    def training_status(self, training_guid):
        status = self.client.training.get_status(training_guid)
        return status
    
    def download_latest_training_log(self, result_bucket, save_filename):
        modified = []
        for item in self.get_bucket_items(result_bucket):
            if item.key.split("/")[-1] == "training-log.txt":
                modified.append((item, item.last_modified))

        tf = sorted(modified, key=lambda x:x[1])[-1][0].key
        print("downloading {} to {}... ".format(tf, save_filename), end="")
        self.cos.Bucket(result_bucket).download_file(tf, save_filename)
        print("done")
    
    def download_latest_model(self, result_bucket, save_filename):
        modified = []
        for item in self.get_bucket_items(result_bucket):
            if item.key.split("/")[-1] == "model.h5":
                modified = [(item, item.last_modified)]

        model_file = sorted(modified, key=lambda x:x[1])[-1][0].key
        print("downloading {} to {}... ".format(model_file, save_filename), end="")
        self.cos.Bucket(result_bucket).download_file(model_file, save_filename)
        print("done")
        
    def download_best_checkpoint(self, result_bucket, save_filename, reverse=False):
        loss = []
        for item in self.get_bucket_items(result_bucket):
            model_name = item.key.split("/")[-1]
            model_loss = model_name.split("-")[-1][:-3]
            if model_name.split("-")[0] == "model":
                loss.append((item, float(model_loss)))

        best_model = sorted(loss, key=lambda x:x[1])[-1]
        model_file, best_loss = best_model[0].key, best_model[1]
        print("downloading {} to {}... ".format(model_file, save_filename), end="")
        self.cos.Bucket(result_bucket).download_file(model_file, save_filename)
        print("done")