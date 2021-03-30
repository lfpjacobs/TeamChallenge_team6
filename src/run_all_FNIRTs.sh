#!/bin/bash
################################################################################
#                              run_all_FNIRTs                                  #
#                                                                              #
# This piece of bash script runs through all subjects and performs a           #
# non-linear registration through FSL's FNIRT algorithm.                       #
#                                                                              #
# Change History                                                               #
# 15/02/2021  Sjors Verschuren      Initial commit   						   #
# 26/02/2021  Sjors Verschuren      Improved results and result structure      #
#                                                                              #
#                                                                              #                                        
################################################################################
################################################################################
################################################################################

## Define variables and files
subjects=( 19 20 23 24 25 26 29 31 32 34 35 61 62 63 69 70 71 72 83 84 89 96 )
dataDir="data/raw"
resultDir="data/FSL_results"

## Make a directory for the results
[ ! -d "$resultDir" ] && mkdir "$resultDir"

## Main subject loop
for subject in "${subjects[@]}"
do
	printf "\n--- Starting FSL registration for subject $subject ---\n"
	
	# ######### PATH DEFENITIONS #########

	## Define where to store results
	result_folder="${resultDir}/rat${subject}/"
	[ ! -d "$result_folder" ] && mkdir "$result_folder"
	[ ! -d "${result_folder}raw/" ] && mkdir "${result_folder}raw/"
	[ ! -d "${result_folder}transforms/" ] && mkdir "${result_folder}transforms/"
	
	## Define subject data paths
	subject_prefix="${dataDir}/rat${subject}_"
	# DWI B0 files
	day0_dwib0_n3="${subject_prefix}dwib0_1_n3.nii.gz"
	day4_dwib0_n3="${subject_prefix}dwib0_3_n3.nii.gz"
	day0_dwib0_bet="${subject_prefix}dwib0_1_bet.nii.gz"
	day4_dwib0_bet="${subject_prefix}dwib0_3_bet.nii.gz"	
	# Mask files
	day0_mask="${subject_prefix}adc1f_lesionmask.nii.gz"
	day4_mask="${subject_prefix}t2map_3_cplxfit_lesionmask.nii.gz"	
	# Additional files
	day0_adc="${subject_prefix}adc1f.nii.gz"
	day4_adc="${subject_prefix}adc3f.nii.gz"
	day0_t2map_cplx="${subject_prefix}t2map_1_cplxfit.nii.gz"
	day4_t2map_cplx="${subject_prefix}t2map_3_cplxfit.nii.gz"
	# day0_t2w1_n3="${subject_prefix}t2w1_1_n3.nii.gz"
	# day0_t2w1_bet="${subject_prefix}t2w1_1_bet.nii.gz"
	# day4_t2w1_n3="${subject_prefix}t2w1_3_n3.nii.gz"
	# day4_t2w1_n3="${subject_prefix}t2w1_3_bet.nii.gz"
	
	## Define partial subject result paths
	# DWI B0 files
	res_day4_dwib0_n3="${result_folder}raw/dwib0_3_n3"
	res_day4_dwib0_bet="${result_folder}raw/dwib0_3_bet"	
	# Mask files
	res_day4_mask="${result_folder}raw/t2map_3_cplxfit_lesionmask"	
	# Additional files
	res_day4_adc="${result_folder}raw/adc3f"
	res_day4_t2map_cplx="${result_folder}raw/t2map_3_cplxfit"
	# res_day4_t2w1_n3="${result_folder}raw/t2w1_3_n3"
	# res_day4_t2w1_n3="${result_folder}raw/t2w1_3_bet"
	
	# ########## CHECKING FILES ##########
	
	## Check for the existence of all strictly necessary files
	printf "\n ------------------------------------------------------------------------------- \n"
	printf "Day 0 DWI n3 image:\t${day0_dwib0_n3}\t\t\t- "
	[ -f "$day0_dwib0_n3" ] && printf "Exists!\n" || printf "Doesn't exist!\n"
	printf "Day 0 DWI bet image:\t${day0_dwib0_bet}\t\t- "
	[ -f "$day0_dwib0_bet" ] && printf "Exists!\n" || printf "Doesn't exist!\n"
	printf "Day 4 DWI n3 image:\t${day4_dwib0_n3}\t\t\t- "
	[ -f "$day4_dwib0_n3" ] && printf "Exists!\n" || printf "Doesn't exist!\n"
	printf "Day 4 DWI bet image:\t${day4_dwib0_bet}\t\t- "
	[ -f "$day4_dwib0_bet" ] && printf "Exists!\n" || printf "Doesn't exist!\n"
	printf "Day 0 lesion mask:\t${day0_mask}\t\t- "
	[ -f "$day0_mask" ] && printf "Exists!\n" || printf "Doesn't exist!\n"
	printf "Day 4 lesion mask:\t${day4_mask}\t- "
	[ -f "$day4_mask" ] && printf "Exists!\n" || printf "Doesn't exist!\n"
	printf " ------------------------------------------------------------------------------- \n"
	
	# ##### PERFORMING REGISTRATIONS #####

	## Perform the initial linear transform (flirt) on the skull-stripped image and apply the transformation to the other images
	printf "\nPerforming initial linear transform (flirt)... "
	flirt -in "$day4_dwib0_bet" -ref "$day0_dwib0_bet" -out "${res_day4_dwib0_bet}_flirt.nii.gz" -omat "${result_folder}transforms/flirt_transformation.mat" -dof 6
	printf "Completed!\n"
	
	printf "Applying transformation on additional images... "
	flirt -in "$day4_dwib0_n3" -ref "$day0_dwib0_bet" -out "${res_day4_dwib0_n3}_flirt.nii.gz" -init "${result_folder}transforms/flirt_transformation.mat" -applyxfm
	flirt -in "$day4_mask" -ref "$day0_mask" -out "${res_day4_mask}_flirt.nii.gz" -init "${result_folder}transforms/flirt_transformation.mat" -applyxfm
	flirt -in "$day4_t2map_cplx" -ref "$day0_t2map_cplx" -out "${res_day4_t2map_cplx}_flirt.nii.gz" -init "${result_folder}transforms/flirt_transformation.mat" -applyxfm
	applyxfm4D "$day4_adc" "$day0_adc" "${res_day4_adc}_flirt.nii.gz" "${result_folder}transforms/flirt_transformation.mat" -singlematrix 2>/dev/null # Different command because of 4D data
	printf "Completed!\n"
	
	## Perform the actual nonlinear transform (fnirt) on the n3-images and apply the deformation field to the other images
	printf "\nPerforming actual nonlinear transform (fnirt)... "
	fnirt --ref="$day0_dwib0_n3" --in="$day4_dwib0_n3" --aff="${result_folder}transforms/flirt_transformation.mat" --cout="${result_folder}/transforms/fnirt_deformation.nii.gz" --iout="${res_day4_dwib0_n3}_fnirt.nii.gz" 2>/dev/null
	printf "Completed!\n"
	
	printf "Applying deformation field on additional images... "
	applywarp --ref="$day0_dwib0_bet" --in="$day4_dwib0_bet" --out="${res_day4_dwib0_bet}_fnirt.nii.gz" --warp="${result_folder}transforms/fnirt_deformation.nii.gz" --premat="${result_folder}transforms/flirt_transformation.mat"
	applywarp --ref="$day0_mask" --in="$day4_mask" --out="${res_day4_mask}_fnirt.nii.gz" --warp="${result_folder}transforms/fnirt_deformation.nii.gz" --premat="${result_folder}transforms/flirt_transformation.mat"
	applywarp --ref="$day0_t2map_cplx" --in="$day4_t2map_cplx" --out="${res_t2map_cplx}_fnirt.nii.gz" --warp="${result_folder}transforms/fnirt_deformation.nii.gz" --premat="${result_folder}transforms/flirt_transformation.mat"
	printf "Completed!\n"
	
	# ###### RESTRUCTURING RESULTS ######
	
	## Extract and copy all relevant images for further research
	printf "\nRestructuring result files... "
	cp "${res_day4_mask}_flirt.nii.gz" "${result_folder}mask_flirt.nii.gz"
	cp "${res_day4_mask}_fnirt.nii.gz" "${result_folder}mask_fnirt.nii.gz"
	cp "${res_day4_dwib0_n3}_flirt.nii.gz" "${result_folder}n3_flirt.nii.gz"
	cp "${res_day4_dwib0_n3}_fnirt.nii.gz" "${result_folder}n3_fnirt.nii.gz"
	cp "${res_day4_dwib0_bet}_flirt.nii.gz" "${result_folder}bet_flirt.nii.gz"
	cp "${res_day4_dwib0_bet}_fnirt.nii.gz" "${result_folder}bet_fnirt.nii.gz"
	printf "Completed!\n"
	
	printf "\nCompleted subject!\n"
done