python methods/deepLearning_singleBin.py --saveFastdatadir files/K562/ -b 0 -i /home/p/palandt/NEAP/backup/outputfiles/K562_scaled.txt -l labelsK562.txt --outputtag K562 --outputfile deepLearningBins.txt
echo "K562"
for i in {1..159}
	do
		python methods/deepLearning_singleBin.py --fastdatadir files/K562 -b $i --outputtag K562 --outputfile deepLearningBins.txt
done

python methods/deepLearning_singleBin.py --saveFastdatadir files/Endo -b 0 -i /home/p/palandt/NEAP/backup/outputfiles/Endo_scaled.txt -l labelsK562.txt --outputtag Endo --outputfile deepLearningBins.txt
echo "Endo"
for i in {1..159}
	do
		python methods/deepLearning_singleBin.py --fastdatadir files/Endo -b $i --outputtag Endo --outputfile deepLearningBins.txt
done

python methods/deepLearning_singleBin.py --saveFastdatadir files/Kera -b 0 -i /home/p/palandt/NEAP/backup/outputfiles/Kera_scaled.txt -l labelsK562.txt --outputtag Kera --outputfile deepLearningBins.txt
echo "Kerac"
for i in {1..159}
	do
		python methods/deepLearning_singleBin.py --fastdatadir files/Kera -b $i --outputtag Kera --outputfile deepLearningBins.txt
done


