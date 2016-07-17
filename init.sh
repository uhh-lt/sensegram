mkdir model
mkdir intermediate

cd word2vec_c
make
cd ..

cd chinese-whispers && mvn package shade:shade
java -cp target/chinese-whispers.jar de.tudarmstadt.lt.wsi.WSI