# decay mode
TRAIN_DIR=$1
NEPOCH=$2
./lwtnn/converters/kerasfunc2json.py \
    saved/dsnn/${TRAIN_DIR}/architecture.json \
    saved/dsnn/${TRAIN_DIR}/weights-${NEPOCH}.h5 \
    data/json/decaymode/variables.json > saved/dsnn/${TRAIN_DIR}/weights-for-lwtnn.json
