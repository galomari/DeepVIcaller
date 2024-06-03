DeepVIcaller

A deep learning model to predict viral integration sites using high-throughput sequencing data. Prerequisites:

• Python (3.7) • TensorFlow (1.13.1): An open-source software library for numerical computation of high performance. TensorFlow • Keras (2.2.4): An Application Programming Interface (API) for deep learning in Python. Keras • Scikit-learn (0.24): A free software library for machine learning in Python. Scikit-learn • PyCharm: A professional Python Integrated Development Environment (IDE). PyCharm • Anaconda (1.9.7): An open-source Python distribution. Anaconda • CUDA (10.0.130): A computing platform launched by graphics card manufacturer NVIDIA. CUDA Toolkit • CuDNN (7.0): A GPU accelerating library for deep neural networks. CuDNN Among them, TensorFlow is equipped with a GPU version, which can be used for accelerated calculations with CUDA and CuDNN. Details Most of the above can be installed by pip with version number, e.g. pip install tensorflow==1.13.1

File Description: • VIcallerData: The folder to store the data. o bam_Test_Data.npy: Test data for BAM files. o bam_label_Test_Label.npy: Labels for BAM test data. o fastqlable_Test_Label.npy: Labels for FASTQ test data. o fatq_Test_Data.npy: Test data for FASTQ files. o vicaller_bam.npy: BAM data matrix. o vicaller_fastq.npy: FASTQ data matrix. • Model: The folder to store the trained DeepVIcaller model. • Test_Result: The folder to store test results of the model.

o vicaller_Result  model_test_results_bam.csv: Test results for BAM data.  model_test_results_fastq.csv: Test results for FASTQ data.

• data: The folder containing input data files. o seq_RNA.bam: Example BAM file. o test2.fastq: Example FASTQ file.

• process: The folder containing data processing scripts. o data_process.py: Script for processing data. o data_processing_updated.py: Updated script for data processing.

• DeepCaller.py: This is the main program, including the creation, loading, and testing of the model. • results_explained.txt: File explaining the test results.

Run:

Run data_process.py first for data encoding. Two files (vicaller_fastq.npy and vicaller_bam.npy) will be generated and stored in the folder VIcallerData.
Then run DeepCaller.py. The test results will be generated and stored in the folder Test_Result.
Contact: If you have any questions, please contact me. Email: galomari@csu.edu
