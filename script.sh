#!/bin/sh
for i in {1..10}
    do
        python cartPole_target.py
        python newCartpole.py
        python cartPole_multi.py
    done

