#!\bin\bash

for low in 10 15 20 25 30 25 40 45 50 55 60; do
  for high in 90 100 110 120 130 140 150 160 170 180 190 200; do
  	echo "${low}" "${high}"
    python3 run_signal_pipeline.py --sma-low "$low" --sma-high "$high" wse_stocks/*.txt
  done
done
