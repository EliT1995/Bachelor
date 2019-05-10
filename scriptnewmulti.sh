#!/bin/sh
for i in {1..10}
    do
        python cartPole_noTarget.py
        python newCartpole.py
        python cartPole_simpleTarget.py
    done

