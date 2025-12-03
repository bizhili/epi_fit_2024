echo "python run.py --epoches 150000 --modelLoad AA --weightModel identical --randomGraph BA                --seed 10 --strains 4  --dense 0 --n 100 --intense 0 --CMDprogress_0_7"
python run.py --epoches 150000 --modelLoad AA --weightModel identical --randomGraph BA                --seed 10 --strains 4  --dense 0 --n 100 --intense 0 --CMDprogress_0_7 &
echo "python run.py --epoches 150000 --modelLoad AA --weightModel identical --randomGraph ER                --seed 11 --strains 4  --dense 0 --n 100 --intense 0 --CMDprogress_1_7"
python run.py --epoches 150000 --modelLoad AA --weightModel identical --randomGraph ER                --seed 11 --strains 4  --dense 0 --n 100 --intense 0 --CMDprogress_1_7 &
echo "python run.py --epoches 150000 --modelLoad AA --weightModel identical --randomGraph RGG                --seed 10 --strains 4  --dense 0 --n 100 --intense 0 --CMDprogress_2_7"
python run.py --epoches 150000 --modelLoad AA --weightModel identical --randomGraph RGG                --seed 10 --strains 4  --dense 0 --n 100 --intense 0 --CMDprogress_2_7 &
echo "python run.py --epoches 150000 --modelLoad AA --weightModel identical --randomGraph WS                --seed 11 --strains 4  --dense 0 --n 100 --intense 0 --CMDprogress_3_7"
python run.py --epoches 150000 --modelLoad AA --weightModel identical --randomGraph WS                --seed 11 --strains 4  --dense 0 --n 100 --intense 0 --CMDprogress_3_7 &
wait
echo "python run.py --epoches 150000 --modelLoad AA --weightModel identical --randomGraph WS                --seed 10 --strains 4  --dense 0 --n 100 --intense 0 --CMDprogress_4_7"
python run.py --epoches 150000 --modelLoad AA --weightModel identical --randomGraph WS                --seed 10 --strains 4  --dense 0 --n 100 --intense 0 --CMDprogress_4_7 &
echo "python run.py --epoches 150000 --modelLoad AA --weightModel identical --randomGraph BA                --seed 11 --strains 4  --dense 0 --n 100 --intense 0 --CMDprogress_5_7"
python run.py --epoches 150000 --modelLoad AA --weightModel identical --randomGraph BA                --seed 11 --strains 4  --dense 0 --n 100 --intense 0 --CMDprogress_5_7 &
echo "python run.py --epoches 150000 --modelLoad AA --weightModel identical --randomGraph ER                --seed 10 --strains 4  --dense 0 --n 100 --intense 0 --CMDprogress_6_7"
python run.py --epoches 150000 --modelLoad AA --weightModel identical --randomGraph ER                --seed 10 --strains 4  --dense 0 --n 100 --intense 0 --CMDprogress_6_7 &
echo "python run.py --epoches 150000 --modelLoad AA --weightModel identical --randomGraph RGG                --seed 11 --strains 4  --dense 0 --n 100 --intense 0 --CMDprogress_7_7"
python run.py --epoches 150000 --modelLoad AA --weightModel identical --randomGraph RGG                --seed 11 --strains 4  --dense 0 --n 100 --intense 0 --CMDprogress_7_7 &
wait
