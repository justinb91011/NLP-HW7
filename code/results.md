python3 tag.py ../data/icdev --train ../data/icsup --crf --rnn_dim 2 --eval_interval 200 --max_steps 6000
Tagging accuracy: all: 66.667%, known: 66.667%, seen: nan%, novel: nan%

python3 tag.py ../data/icdev --train ../data/icsup --crf --rnn_dim 5 --eval_interval 200 --max_steps 6000
Tagging accuracy: all: 78.788%, known: 78.788%, seen: nan%, novel: nan

python3 tag.py ../data/icdev --train ../data/icsup --crf --rnn_dim 10 --eval_interval 200 --max_steps 6000
Best seen: Tagging accuracy: all: 84.848%, known: 84.848%, seen: nan%, novel: nan%



python3 tag.py ../data/endev --train ../data/ensup-tiny --crf --rnn_dim 2 --eval_interval 200 --max_steps 4000
INFO:eval:Cross-entropy: 2.1331 nats (= perplexity 8.441)
Tagging accuracy: all: 44.996%, known: 85.675%, seen: nan%, novel: 6.321%

python3 tag.py ../data/endev --train ../data/ensup-tiny --crf --rnn_dim 5 --eval_interval 200 --max_steps 4000
INFO:eval:Cross-entropy: 1.4646 nats (= perplexity 4.326)
INFO:eval:Tagging accuracy: all: 54.800%, known: 93.540%, seen: nan%, novel: 17.969%

python3 tag.py ../data/endev --train ../data/ensup-tiny --crf --rnn_dim 12 --eval_interval 200 --max_steps 4000
INFO:eval:Tagging accuracy: all: 48.219%, known: 95.296%, seen: nan%, novel: 3.462%

python3 tag.py ../data/endev --train ../data/ensup-tiny --crf --rnn_dim 10 --eval_interval 200 --max_steps 4000
:Tagging accuracy: all: 65.305%, known: 96.624%, seen: nan%, novel: 35.530%

python3 tag.py ../data/endev --train ../data/ensup-tiny --crf --rnn_dim 10 --eval_interval 400 --max_steps 5000
Tagging accuracy: all: 64.437%, known: 97.027%, seen: nan%, novel: 33.453%

python3 tag.py ../data/endev --train ../data/ensup-tiny --crf --lr .5 --rnn_dim 10 --eval_interval 100 --max_steps 4000
INFO:eval:Tagging accuracy: all: 46.211%, known: 94.654%, seen: nan%, novel: 0.155%