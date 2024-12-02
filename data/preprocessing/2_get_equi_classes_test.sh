for pProb in 0.025 0.075 0.125 0.175 0.225 0.275 0.325 0.375 0.425 0.475 0.525 0.575 0.625 0.675 0.725
do
    for qProb in 0.025 0.075 0.125 0.175 0.225 0.275 0.325 0.375 0.425 0.475 0.525 0.575 0.625 0.675 0.725
    do
        in="../processed/zone/test/1000bp/test_zone_all_1000bp_permuted_p${pProb}_q${qProb}.csv"
        out="../processed/zone/test/1000bp_equis/test_zone_all_1000bp_permuted_equis_p${pProb}_q${qProb}.csv"
        python get_equi_classes.py -i $in -o $out
    done
done
