import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

sagemaker_session = sagemaker.Session()
role = 'arn:aws:iam::372535189767:role/service-role/AmazonSageMakerServiceCatalogProductsUseRole'

# Create a ScriptProcessor
script_processor = ScriptProcessor(
    image_uri='372535189767.dkr.ecr.eu-west-2.amazonaws.com/pl-sagemaker-pipeline:dev', 
    command=['python3'],
    role=role,
    instance_count=1,
    instance_type='ml.t3.xlarge',
    sagemaker_session=sagemaker_session
)

# Run the processing job
script_processor.run(
    code='premier_league/sagemaker_pipeline.py'
)






