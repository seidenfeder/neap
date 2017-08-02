#first of all give the name of the dataset and the name of the two tsv-files (gene expression file) 
name="K562"
tsv1="ENCFF047WAI.tsv"
tsv2="ENCFF937GNL.tsv"

#First get the labels file (median split)
python preprocessing/labelGenes.py -a /home/proj/biocluster/praktikum/neap_pearl/palandt_schmid/files/Data/$name/$tsv1 -b /home/proj/biocluster/praktikum/neap_pearl/palandt_schmid/files/Data/$name/$tsv2 --protCod -o labels$name.txt -g /home/proj/biocluster/praktikum/neap_pearl/palandt_schmid/files/Data/$name/gencode.v26.annotation.gtf 

#get the binning file
python preprocessing/bins_annotated.py -d /home/proj/biocluster/praktikum/neap_pearl/palandt_schmid/files/Data/$name/bigWigs/ -a /home/proj/biocluster/praktikum/neap_pearl/palandt_schmid/files/Data/$name/$tsv1 -b /home/proj/biocluster/praktikum/neap_pearl/palandt_schmid/files/Data/$name/$tsv2 --annot /home/proj/biocluster/praktikum/neap_pearl/palandt_schmid/files/Data/$name/metadata.tsv -g /home/proj/biocluster/praktikum/neap_pearl/palandt_schmid/files/Data/$name/gencode.v26.annotation.gtf --out input$name.txt --protCod
#scale the binning file
python preprocessing/normalizations.py -i input$name.txt -o inputScaled$name.txt -m Scale -n ~/NEAP/Data/$name/metadata.tsv

#get the bin importance for every bin for all methods
python parameterFitting/binImportance.py -i inputScaled$name.txt -l labels$name.txt -m RFC
python parameterFitting/binImportance.py -i inputScaled$name.txt -l labels$name.txt -m SVC
python parameterFitting/binImportance.py -i inputScaled$name.txt -l labels$name.txt -m RFR
python parameterFitting/binImportance.py -i inputScaled$name.txt -l labels$name.txt -m SVR
python parameterFitting/binImportance.py -i inputScaled$name.txt -l labels$name.txt -m LR

#get the performance for all methods with all bins
#regression is without the zero values
python methods/classification.py -i inputScaled$name.txt -l labels$name.txt -c 10 -o classification.txt -a -n -m RF
python methods/classification.py -i inputScaled$name.txt -l labels$name.txt -c 10 -o classification.txt -a -n -m SVM
python methods/regression.py -i inputScaled$name.txt -m RF -n -a -c 10 
python methods/regression.py -i inputScaled$name.txt -m SVM -n -a -c 10 
python methods/regression.py -i inputScaled$name.txt -m LR -n -a -c 10

#ckeck the importance of the histone modifications
python parameterFitting/HisModImportanceSingle.py -i inputScaled$name.txt -l labels$name.txt -c 10 -a -o histModImportanceSingle.txt -n -m RFC
python parameterFitting/HisModImportanceSingle.py -i inputScaled$name.txt -l labels$name.txt -c 10 -a -o histModImportanceSingle.txt -n -m SVC
python parameterFitting/HisModImportanceSingle.py -i inputScaled$name.txt -l labels$name.txt -c 10 -a -o histModImportanceSingle.txt -n -m RFR
python parameterFitting/HisModImportanceSingle.py -i inputScaled$name.txt -l labels$name.txt -c 10 -a -o histModImportanceSingle.txt -n -m SVR
python parameterFitting/HisModImportanceSingle.py -i inputScaled$name.txt -l labels$name.txt -c 10 -a -o histModImportanceSingle.txt -n -m LR
python parameterFitting/HisModImportanceSingle.py -i inputScaled$name.txt -l labels$name.txt -c 10 -a -o histModImportancePairs.txt -n -m RFC
python parameterFitting/HisModImportanceSingle.py -i inputScaled$name.txt -l labels$name.txt -c 10 -a -o histModImportancePairs.txt -n -m SVC
python parameterFitting/HisModImportanceSingle.py -i inputScaled$name.txt -l labels$name.txt -c 10 -a -o histModImportancePairs.txt -n -m RFR
python parameterFitting/HisModImportanceSingle.py -i inputScaled$name.txt -l labels$name.txt -c 10 -a -o histModImportancePairs.txt -n -m SVR
python parameterFitting/HisModImportanceSingle.py -i inputScaled$name.txt -l labels$name.txt -c 10 -a -o histModImportancePairs.txt -n -m LR


#check the performance with differnet labeling methods and different normalization methods
python parameterFitting/evaluateGeneLabeling.py -a /home/proj/biocluster/praktikum/neap_pearl/palandt_schmid/files/Data/$name/$tsv1 -b /home/proj/biocluster/praktikum/neap_pearl/palandt_schmid/files/Data/$name/$tsv2 -o evalLabels$name.txt -g /home/proj/biocluster/praktikum/neap_pearl/palandt_schmid/files/Data/$name/gencode.v26.annotation.gtf -i input$name.txt
python parameterFitting/evaluateNormalization.py -o evalNorm$name.txt -i input$name.txt -l labels$name.txt
