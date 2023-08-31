runs_root="runs/marchenko_pastur/global_empirical"
dump_root="dumps/mp"

for p in $(ls $runs_root/)
do
for run in $(ls "$runs_root/$p/")
do
for r in $(ls "$runs_root/$p/$run/" | grep "ratio")
do

python etc/dump.py --source-dir $runs_root/$p/$run/$r --output-dir $dump_root/$p/$run/$r

done
done
done