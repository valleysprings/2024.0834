# main test
nohup python ../ics/main.py --dataset karate --num_queries 200 --al_round 3 --al_method aggregated --deterministic >> ../log/main_test_core.log 2>&1 & 
nohup python ../ics/main.py --dataset dolphins --num_queries 200 --al_round 3 --al_method aggregated --deterministic >> ../log/main_test_core.log 2>&1 & 
nohup python ../ics/main.py --dataset football --num_queries 200 --al_round 3 --al_method aggregated --deterministic >> ../log/main_test_core.log 2>&1 & 
nohup python ../ics/main.py --dataset eu-core --num_queries 200 --al_round 3 --al_method aggregated --deterministic >> ../log/main_test_core.log 2>&1 & 
nohup python ../ics/main.py --dataset amazon --num_queries 200 --al_round 3 --al_method aggregated --deterministic >> ../log/main_test_core.log 2>&1 & 
nohup python ../ics/main.py --dataset dblp --num_queries 200 --al_round 3 --al_method aggregated --deterministic >> ../log/main_test_core.log 2>&1 & 
nohup python ../ics/main.py --dataset youtube --num_queries 200 --al_round 3 --al_method aggregated --deterministic >> ../log/main_test_core.log 2>&1 & 
nohup python ../ics/main.py --dataset lj --num_queries 200 --al_round 3 --al_method aggregated --deterministic >> ../log/main_test_core.log 2>&1 & 
