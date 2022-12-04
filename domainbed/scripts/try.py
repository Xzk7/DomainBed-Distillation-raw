import sys
sys.path.append("/home/zkxu/DG/code/DomainBed/")

from domainbed.lib import misc, reporting

if __name__ == '__main__':
    records = reporting.load_records("/data1/zkxu/DG/code/DomainBed/VLCS_ERM_R50")

    # hparams_teacher = {'batch_size': 39, 'class_balanced': False, 'data_augmentation': True, 'lr': 2.7028930742148706e-05, 'nonlinear_classifier': False, 'resnet18': False, 'resnet_dropout': 0.5, 'weight_decay': 0.00044832883881609976}
    hparams_teacher = {'batch_size': 32, 'class_balanced': False, 'data_augmentation': True, 'lr': 5e-05, 'nonlinear_classifier': False, 'resnet18': False, 'resnet_dropout': 0.0, 'weight_decay': 0.0}
    records = records.filter(
        lambda r:
        r['args']['dataset'] == "VLCS" and
        r['args']['algorithm'] == "ERM" and
        # r['args']['test_envs'] == [1] and
        r['hparams'] == hparams_teacher
    )
    for record in records:
        print(record['args']['test_envs'])