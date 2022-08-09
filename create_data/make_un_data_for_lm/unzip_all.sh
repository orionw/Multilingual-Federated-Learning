# tar.gz files were split - we need to combine them to read them
cat UNv1.0-TEI.fr.tar.gz.* > UNv1.0-TEI.fr.tar.gz
cat UNv1.0-TEI.ar.tar.gz.* > UNv1.0-TEI.ar.tar.gz
cat UNv1.0-TEI.es.tar.gz.* > UNv1.0-TEI.es.tar.gz
cat UNv1.0-TEI.en.tar.gz.* > UNv1.0-TEI.en.tar.gz
cat UNv1.0-TEI.ru.tar.gz.* > UNv1.0-TEI.ru.tar.gz
# now decode them
tar -xvf UNv1.0-TEI.fr.tar.gz &&
tar -xvf UNv1.0-TEI.en.tar.gz &&
tar -xvf UNv1.0-TEI.ar.tar.gz &&
tar -xvf UNv1.0-TEI.ru.tar.gz &&
tar -xvf UNv1.0-TEI.es.tar.gz &&
tar -xvf UNv1.0-TEI.zh.tar.gz.00