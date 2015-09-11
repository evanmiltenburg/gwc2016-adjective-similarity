# WordNet::Similarity

I used the following two commands to get the desired similarity values.

### Lesk

`perl similarity.pl --type=WordNet::Similarity::lesk --allsenses --file=/path/to/pairs.txt > /path/to/lesk_result.txt`

### Hso

`perl similarity.pl --type=WordNet::Similarity::hso --allsenses --file=/path/to/pairs.txt > /path/to/hso_result.txt`
