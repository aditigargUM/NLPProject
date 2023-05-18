# Add parent directory to python path to access lightning_base.py
#export PYTHONPATH="../":"${PYTHONPATH}"
#export MODEL_NAME_OR_PATH="./examples/research_projects/rag/outputs_14/checkpoint2"
#export OUTPUT_DIR="./examples/research_projects/rag/outputs_16/"
#export DATA_DIR="/work/pi_adrozdov_umass_edu/syerawar_umass_edu/LF/RAG_Finetune/transformers/examples/research_projects/rag/data_msmarco"

# Start a single-node Ray cluster.
ray start --head

# A sample finetuning run, you need to specify data_dir, output_dir and model_name_or_path
# run ./examples/rag/finetune_rag.sh --help to see all the possible options

python /work/pi_adrozdov_umass_edu/syerawar_umass_edu/LF/RAG_Finetune/transformers/examples/research_projects/rag/finetune_rag.py \
    --data_dir /work/pi_adrozdov_umass_edu/syerawar_umass_edu/LF/RAG_Finetune/transformers/examples/research_projects/rag/data_msmarco_full \
    --output_dir /work/pi_adrozdov_umass_edu/syerawar_umass_edu/LF/RAG_Finetune/transformers/examples/research_projects/rag/outputs_19/ \
    --model_name_or_path /work/pi_adrozdov_umass_edu/syerawar_umass_edu/LF/RAG_Finetune/transformers/examples/research_projects/rag/checkpoint2/ \
    --model_type rag_sequence \
    --fp16 \
    --gpus 2 \
    --profile \
    --do_train \
    --do_predict \
    --n_val -1 \
    --train_batch_size 4 \
    --eval_batch_size 1 \
    --max_source_length 128 \
    --max_target_length 128 \
    --val_max_target_length 128 \
    --test_max_target_length 128 \
    --label_smoothing 0.1 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --weight_decay 0.001 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 0.1 \
    --learning_rate 3e-4 \
    --num_train_epochs 10 \
    --warmup_steps 4 \
    --lr_scheduler polynomial \
    --gradient_accumulation_steps 1 \
    --num_retrieval_workers 1 \
    --distributed_retriever ray \
    --index_name custom \
    --passages_path /work/pi_adrozdov_umass_edu/syerawar_umass_edu/LF/RAG_Finetune/transformers/output_marco/my_knowledge_dataset \
    --index_path /work/pi_adrozdov_umass_edu/syerawar_umass_edu/LF/RAG_Finetune/transformers/output_marco/my_knowledge_dataset_hnsw_index.faiss \

# Stop the Ray cluster.
ray stop
