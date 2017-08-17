export BUCKET_NAME=unlabeled_sk_bucket
export JOB_NAME="JOB_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
export REGION=us-east1

gcloud ml-engine jobs submit training $JOB_NAME \
--job-dir gs://$BUCKET_NAME/$JOB_NAME \
--runtime-version 1.0 --module-name trainer.build_autoencoder_gcloud \
--package-path ./trainer \
--region $REGION \
--config=trainer/config.yaml \
-- \
--train-file gs://unlabeled_sk_bucket/input/Unlabeled_X.npy \
--output_dir gs://unlabeled_sk_bucket/output/ \
--learning-rate "0.001" \
--n-epochs "300"