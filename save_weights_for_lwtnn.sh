# decay mode
NEPOCH=$1
./lwtnn/converters/kerasfunc2json.py saved/dsnn/architecture.json saved/dsnn/weights-${NEPOCH}.h5 data/json/decaymode/variables.json > saved/dsnn/weights-for-lwtnn.json
