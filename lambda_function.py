import os
import tarfile
import json
import boto3
import sys
import uuid
from datetime import datetime

resource_bucket_name = 'case-study-resource' # replace with your resource bucket name
pkg_name = 'pkg_tmp.tar.gz' # replace with your pkg_tmp file
model_params = 'resnet50_v2.params' # replace with your model file
output_bucket = 'case-study-output' # replace with your output_bucket name
lable_classes = 'synset.txt' # replace with your synset file name

# Download package from s3 bucket
pkg_path = '/tmp/{}'.format(pkg_name)
s3 = boto3.resource('s3')
s3.Bucket(resource_bucket_name).download_file(pkg_name, pkg_path)

# cache work directory
work = os.getcwd()
# change to /tmp to extract package
os.chdir('/tmp')
tf = tarfile.open(pkg_path)
tf.extractall()
# remove package.tar.gz file
os.remove(pkg_path)
# add /tmp to python path
sys.path.insert(0, '/tmp/')
# go back to lambda work directory
os.chdir(work)

import mxnet as mx
from mxnet import image
from mxnet import nd
from mxnet.gluon.model_zoo import vision as models

# create model and load parameter
params = '/tmp/{}'.format(model_params)
net = models.resnet50_v2(pretrained=False)
s3.Bucket(resource_bucket_name).download_file(model_params, params)
net.load_parameters(params, ctx=mx.cpu())
net.hybridize()

# generate lable_list
with open(lable_classes, 'r') as f:
    labels = [' '.join(l.split()[1:]) for l in f]

def lambda_handler(event, context):
    """
    Main entry point for AWS Lambda. It parse the input event, transform the image, 
    make prediction and finally store the result into output_bucket.
    """

    def transform_image(img_path):
        '''
        Image transformation for model trained on ImageNet dataset.
        '''
        img = image.imread(img_path)
        data = image.resize_short(img, 256)
        data, _ = image.center_crop(data, (224,224))
        data = data.transpose((2,0,1)).expand_dims(axis=0)
        rgb_mean = nd.array([0.485, 0.456, 0.406]).reshape((1,3,1,1))
        rgb_std = nd.array([0.229, 0.224, 0.225]).reshape((1,3,1,1))
        data = (data.astype('float32') / 255 - rgb_mean) / rgb_std
        return data
    
    
    for record in event['Records']:
        # retrieve input image from Lambda event
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        img_path = '/tmp/{}'.format(key)
        s3.Bucket(bucket).download_file(key, img_path)
    
        # transform image and perform inference
        data = transform_image(img_path)
        predict = net(data)
        idx = predict.topk(k=1)[0]
        idx = int(idx.asscalar())
        os.remove(img_path)
        
        # create result file and send it to output bucket
        # the title of the file contains the image file name and the time it was created
        # the content of the file contains the predicted class.
        time = datetime.now().strftime("%d%m%Y-%H:%M:%S")
        file_name = '{}_{}.txt'.format(key, time)
        content=labels[idx]
        s3.Object(output_bucket, file_name).put(Body=content)
        
    
    return {
        "statusCode": 200,
        "body": json.dumps('Successfully classified an image')
    }