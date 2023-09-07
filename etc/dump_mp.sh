runs_root="runs/marchenko_pastur/lr_epoch100"
dump_root="dumps/mp/lr_epoch100"

for wd in $(ls $runs_root/)
do
for p in $(ls $runs_root/$wd)
do
for run in $(ls "$runs_root/$wd/$p/")
do

ls "$runs_root/$wd/$p/$run/" | grep "ratio" | xargs -I {} -P 5 python etc/dump.py --source-dir $runs_root/$wd/$p/$run/{} --output-dir $dump_root/$wd/$p/$run/{}

done
done
done
