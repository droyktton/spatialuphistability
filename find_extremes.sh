for f in $1/rk4_k_sweep.txt
do 
#	echo $(gawk 'BEGIN{kmax=0;hmax=0;maxs=0;}{max=($3>$4)?($3):($4); maxs=(max>maxs)?(max):(maxs); if(max>1 && $1>kmax) kmax=$1; if(max>1 && $2>hmax) hmax=$2 }END{print kmax, hmax, maxs}' $f) $f 
	echo $(gawk 'BEGIN{kmax=0;hmax=0;maxs=0;}{max=($3>$4)?($3):($4); maxs=(max>maxs)?(max):(maxs); if(max>1 && $1>kmax) kmax=$1; if(max>1 && $2>hmax) hmax=$2 }END{print maxs}' $f) $f 
done
