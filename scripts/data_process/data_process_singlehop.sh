WORK_DIR=directory-to-code/TraceCredit-RL
LOCAL_DIR=directory-to-data

## process multiple dataset search format train file
DATA=nq
python $WORK_DIR/scripts/data_process/qa_search_train_merge.py --local_dir $LOCAL_DIR --data_sources $DATA

## process multiple dataset search format test file
DATA=nq,triviaqa,popqa
python $WORK_DIR/scripts/data_process/qa_search_test_merge.py --local_dir $LOCAL_DIR --data_sources $DATA
