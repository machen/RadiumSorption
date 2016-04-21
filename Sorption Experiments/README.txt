Folder is a compilation of data and scripts used during radium sorption isotherm experiments

Workflow:
1) Collect experiment data, create new folder for experiment
2) Make copy of template file in folder, add data as specified. Be sure sample names contain overlapping elements, and that the parameter spreadsheet is correctly filled in.
3) Run Radium Sample Analysis v2.py on excel file, generating result file.
4) Manually verify data, add to file Sorption Experiment Master Table.xlsx, filling in all data fields
5) Plot and interact with all data using MasterTableAccessScript.py

Scripts:

MasterTableAccessScript.py: Script that draws data from the master table. There are multiple functions that allow for plotting of pH data vs Sorbed Concentrations or Isotherm data.

pH Sweep Analysis.py: Legacy script for plotting and analyzing pH sweep data. Supplanted by master table access script.

Radium Sample Analysis v2.py: Python script that takes in sorption experiment data, and converts the scintillation counter results to actual activities of radium, then calculating the sorbed fractions, and saving them to a results file. These should be not be rerun unless absolutely necessary, as they represent the source of data. All calculations are done here, as opposed to in the master

Radium Sample Analysis v3.py: new version python script that accommodates experiments that take in measurements of both solid and liquid radiation counts

Radium Sorption Analysis.py: Legacy script used for plotting and analyzing sorption isotherms. Supplanted by master table access script.

Radium Stock Calculation.py: Legacy script that was used to check radium stock concentrations and establish a calibration for the scintillation counter.

Files:

ExperimentDataTemplate v3.xlsx: Data template for sorption experiment data script version 3. A ton of sheets.

Sorption Experiment Master Table.xlsx: Master table of experimental data. All parameters should be in this table, allowing for easy plotting and analysis.