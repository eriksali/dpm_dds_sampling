DEVICES='0'


# ImageNet256 with classifier guidance (large guidance scale) example

data="imagenet256_guided"
scale="8.0"
sampleMethod='dpmsolver++'
type="dpmsolver"
steps="20"
##DIS="time_uniform"
DIS="logSNR"
order="2"
method="multistep"
##method="singlestep"

workdir="experiments/"$data"/"$sampleMethod"_"$method"_order"$order"_"$steps"_"$DIS"_scale"$scale"_type-"$type"_thresholding"
CUDA_VISIBLE_DEVICES=$DEVICES python ddpm.py --config $data".yml" --exp=$workdir --sample --timesteps=$steps --eta 0 --ni --skip_type=$DIS --sample_type=$sampleMethod --dpm_solver_order=$order --dpm_solver_method=$method --dpm_solver_type=$type --port 12350 --scale=$scale --thresholding
