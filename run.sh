#!\bin\bash

for ((low=5; low<=70; low+=5)); do
  for ((high=80; high<=200; high+=5)); do
  	echo "${low}" "${high}"
    python3 run_signal_pipeline.py --sma-low "$low" --sma-high "$high" wse_stocks/*.txt
  done
done
