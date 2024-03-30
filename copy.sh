source_dir="/public/home/zhoujunbao/VOST-code-2"


cp -r ${source_dir}/*.md ./

cp -r ${source_dir}/evaluation ./

mkdir aot_plus
cp -r ${source_dir}/aot_plus/configs ./aot_plus
cp -r ${source_dir}/aot_plus/dataloaders ./aot_plus
cp -r ${source_dir}/aot_plus/networks ./aot_plus
cp -r ${source_dir}/aot_plus/tools ./aot_plus
cp -r ${source_dir}/aot_plus/utils ./aot_plus

# scp ${source_dir}/aot_plus/*.py ./aot_plus
cp ${source_dir}/aot_plus/*.sh ./aot_plus