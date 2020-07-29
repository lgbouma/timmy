source activate py37;

python plot_scene.py
python plot_hr.py
python plot_full_kinematics.py
python plot_rotation.py
python plot_lithium.py
python plot_rp_vs_age_scatter.py
python plot_fpscenarios.py

echo "WRN! Did not run plot_rvs.py, because requires env py37_emcee2"

echo "WRN! Running fit_tessindivtransit and fit_allindivtransit, assuming the chains already exist"

python fit_tessindivtransit.py
python fit_allindivtransit.py
