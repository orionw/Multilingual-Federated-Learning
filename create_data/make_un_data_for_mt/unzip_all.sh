# tar.gz files were split - we need to combine them to read them
cat UNv1.0.ar-es.tar.gz.* > UNv1.0-TEI.ar-es.tar.gz
cat UNv1.0.ru-zh.tar.gz.* > UNv1.0-TEI.ru-zh.tar.gz
cat UNv1.0.en-fr.tar.gz.* > UNv1.0-TEI.en-fr.tar.gz
# now decode them
tar -xvf UNv1.0-TEI.en-fr.tar.gz &&
tar -xvf UNv1.0-TEI.ar-es.tar.gz &&
tar -xvf UNv1.0-TEI.ru-zh.tar.gz 
mkdir splits_data