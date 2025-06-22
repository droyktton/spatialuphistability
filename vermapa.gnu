set terminal pngcairo enhanced font "Helvetica,14" size 600,500
set output 'heatmap.png'

set xlabel "H/H_W"
set ylabel "c^{1/2} {/Symbol D}  k"
set zlabel "log(max floquet multiplier)" 
 
set view map
set pm3d map
set pm3d implicit  # assumes grid structure from x/y but data is unsorted
set cbrange [0:10]

tit = sprintf("{/Symbol a}=%.2f",alpha)

set title tit
splot [1:7.5][0:1.5] 'rk4_k_sweep.txt' using 2:1:(log(($3>$4)?(($3>1)?($3):(0)):(($4>1)?($4):(0)))) with pm3d notitle

#splot 'data.txt' using 2:1:(log($3)) with pm3d notitle

#movie
#ffmpeg -framerate 30 -pattern_type glob -i 'run_alpha*/heat*.png' -c:v libx264 -pix_fmt yuv420p output.mp4
