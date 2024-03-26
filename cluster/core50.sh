# Setup
DIR="$( cd "$( dirname "$0" )" && pwd )"
mkdir $DIR/../work/data/core50
wget --directory-prefix=$DIR'/../work/data/core50' http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip
