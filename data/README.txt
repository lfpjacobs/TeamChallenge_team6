Here, the data used for this project is stored.
Proper file structure is provided below:

 
data --- FSL_results / * 	      % Here, we store the results from the FSL run.
      |                           They may be found by running the run_all_FNIRTs.sh bash script
      -- preprocessed / *       % Here, we store the preprocessed data used for the cGAN.
      |                           This folder is produced by running preprocessing.py
      -- raw / *                % Here, we store the raw data. 
      |                           It may be found in the TEAMS files section
      -- referencecat / *       % Here, we store the reference images.
      |                           They may also be found in the TEAMS files section.
      -- responcevecs.xlsx      % Excel file that contains the physical metrics per subject
                                  This may also be found in the TEAMS files section (responsevecs_TC20analysis210113.xlsx)
