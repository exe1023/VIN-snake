.PHONY: deps test train play play-gui play-human

LEVEL="snakeai/levels/10x10-blank.json"
MODEL="dqn-final.model"
ATTENTION=1

deps:
	python3.6 -m pip install --upgrade -r requirements.txt

test:
	PYTHONPATH=$(PYTHONPATH):. py.test snakeai/tests

train:
	./train.py --level $(LEVEL) --num-episodes 30000 --attention $(ATTENTION)

play:
	./play.py --interface cli --agent dqn --model $(MODEL) --level $(LEVEL) --num-episodes 100 --attention $(ATTENTION)

play-gui:
	./play.py --interface gui --agent dqn --model $(MODEL) --level $(LEVEL) --num-episodes 1 --attention $(ATTENTION)

play-human:
	./play.py --interface gui --agent human --level $(LEVEL) --num-episodes 1 --attention $(ATTENTION)
