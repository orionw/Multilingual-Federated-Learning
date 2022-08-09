for lang in fr ja
do
    for split in train valid test
    do
        src=en
        tgt=$lang
        input=MTNT/$split/$split.en-$lang.tsv
        output=no-punct.en-$lang
        python remove_too_much_punc.py --input <(gzip -c $input) --bitext $output --src-lang $src --tgt-lang $tgt
        paste $output.$src $output.$tgt | cat -n > $input.corrected
    done
done