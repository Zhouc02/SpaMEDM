def params(args):
    if args.data_type == 'HLN_A1':
        args.n_clusters = 10
        args.mask = 0.50
        args.weight1, args.weight2, args.weight3, args.weight4 = 15, 0.01, 0.5, 2
        args.random_seed = 2026
        args.pruning = True
        args.file_fold = '/root/PRAGA/Data/HLN/'

    elif args.data_type == 'HLN_D1':
        args.n_clusters = 10
        args.mask = 0.50
        args.weight1, args.weight2, args.weight3, args.weight4 = 15, 0.5, 0.5, 0.5
        args.random_seed = 2024
        # args.mask = 0.25
        # args.weight1, args.weight2, args.weight3, args.weight4 = 15, 0.01, 0.5, 1
        args.pruning = True
        args.file_fold = '/remote-home/zhouchang/PRAGA/Data/other_data/Data/HLN_D1/'

    elif args.data_type == 'E18_5-S1':
        args.n_clusters = 14
        args.mask = 0.25
        args.weight1, args.weight2, args.weight3, args.weight4 = 10, 0.01, 0.5, 1
        # args.weight1, args.weight2, args.weight3, args.weight4 = 15, 0.01, 1, 2
        args.random_seed = 2023
        args.file_fold = '/remote-home/zhouchang/PRAGA/Data/other_data/Data/E18_5-S1/'

    elif args.data_type == 'E15_5-S1':
        args.n_clusters = 12
        args.mask = 0.25
        args.weight1, args.weight2, args.weight3, args.weight4 = 5, 0.01, 0.5, 0.5
        # args.weight1, args.weight2, args.weight3, args.weight4 = 15, 0.01, 1, 1
        args.file_fold = '/remote-home/zhouchang/PRAGA/Data/other_data/Data/E15_5-S1/'

    elif args.data_type == 'E13_5-S1':
        args.n_clusters = 12
        args.mask = 0.50
        args.weight1, args.weight2, args.weight3, args.weight4 = 15, 0.01, 0.1, 2
        args.random_seed = 2025
        args.file_fold = '/remote-home/zhouchang/PRAGA/Data/other_data/Data/E13_5-S1/'

    elif args.data_type == 'E11_0-S1':
        args.n_clusters = 8
        args.mask = 0.25
        args.weight1, args.weight2, args.weight3, args.weight4 = 15, 0.01, 1, 0.5
        args.random_seed = 2022
        args.file_fold = '/remote-home/zhouchang/PRAGA/Data/other_data/Data/E11_0-S1/'

    elif args.data_type == 'Slide':
        args.n_clusters = 10
        args.mask = 0.50
        args.weight1, args.weight2, args.weight3, args.weight4 = 15, 0.01, 1, 0.5
        args.random_seed = 2025
        args.pruning = True
        args.file_fold = '/root/PRAGA/Data/Slide_tag/'

    elif args.data_type == 'meta':
        args.n_clusters = 11
        args.mask = 0.25
        args.weight1, args.weight2, args.weight3, args.weight4 = 5, 0.05, 0.5, 0.5
        # args.weight1, args.weight2, args.weight3, args.weight4 = 15, 0.01, 1, 1
        args.file_fold = '/root/PRAGA/Data/mouse_brain_RNA_meta/'

    elif args.data_type == 'CITE':
        args.n_clusters = 8
        args.mask = 0.25
        args.weight1, args.weight2, args.weight3, args.weight4 = 2, 0.01, 1, 0.5
        # args.weight1, args.weight2, args.weight3, args.weight4 = 15, 0.01, 1, 1
        args.file_fold = '/root/PRAGA/Data/COSMOS/'

    elif args.data_type == 'Mouse_RNA_H3K27ac':
        args.n_clusters = 18
        args.mask = 0.25
        args.weight1, args.weight2, args.weight3, args.weight4 = 15, 0.01, 0.5, 1
        args.single = True
        args.file_fold = '/root/PRAGA/Data/total_data/Mouse_Brain_H3K27ac/'

    elif args.data_type == 'Mouse_RNA_H3K4me3':
        args.n_clusters = 18
        args.mask = 0.25
        args.weight1, args.weight2, args.weight3, args.weight4 = 15, 0.01, 0.1, 1
        args.single = True
        args.file_fold = '/root/PRAGA/Data/total_data/Mouse_Brain_H3K4me3/'

    elif args.data_type == 'Mouse_RNA_H3K27me3':
        args.n_clusters = 18
        args.mask = 0.25
        args.weight1, args.weight2, args.weight3, args.weight4 = 15, 0.01, 0.1, 0.5
        args.single = True
        args.file_fold = '/root/PRAGA/Data/total_data/Mouse_Brain_H3K27me3/'

    elif args.data_type == 'Mouse_RNA_ATAC':
        args.n_clusters = 18
        args.mask = 0.25
        args.weight1, args.weight2, args.weight3, args.weight4 = 15, 0.01, 0.1, 0.5
        args.single = True
        args.file_fold = '/root/PRAGA/Data/total_data/Mouse_Brain_ATAC/'

    elif args.data_type == 'SPOTS1':
        args.n_clusters = 5
        args.mask = 0.50
        args.weight1, args.weight2, args.weight3, args.weight4 = 15, 0.01, 1, 0.5
        args.file_fold = '/root/PRAGA/Data/total_data/Mouse_Spleen1/'

    elif args.data_type == 'SPOTS2':
        args.n_clusters = 5
        args.mask = 0.50
        args.weight1, args.weight2, args.weight3, args.weight4 = 15, 0.01, 1, 0.5
        args.file_fold = '/root/PRAGA/Data/total_data/Mouse_Spleen2/'

    return args
