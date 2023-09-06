runs_root="runs/marchenko_pastur/lr"
dump_root="dumps/mp/lr"

for wd in $(ls $runs_root/)
do
for p in $(ls $runs_root/$wd)
do
for run in $(ls "$runs_root/$wd/$p/")
do
for r in $(ls "$runs_root/$wd/$p/$run/" | grep "ratio")
do

python etc/dump.py --source-dir $runs_root/$wd/$p/$run/$r --output-dir $dump_root/$wd/$p/$run/$r

done
done
done
done
