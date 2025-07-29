#!\bin\bash

for low in 10 20 30 40 50; do
  for high in 100 120 140 160 180 200; do
  	echo "${low}" "${high}"
    python3 run_signal_pipeline.py --sma-low "$low" --sma-high "$high" wse_stocks/*.txt
  done
done
