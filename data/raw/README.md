# Dados

Fonte Kaggle: [Airbnb Recruiting New User Bookings](https://www.kaggle.com/competitions/airbnb-recruiting-new-user-bookings).

Arquivos esperados nesta pasta:

- `train_users_2.csv`
- `test_users.csv`
- `sessions.csv`
- `countries.csv`
- `age_gender_bkts.csv`
- `sample_submission_NDF.csv`

- O fluxo combina dados de usuarios, sessoes e tabelas auxiliares do desafio.
- O arquivo `sessions.csv` e grande; para GitHub, prefira baixa-lo no Colab em vez de versiona-lo.

## Download via Kaggle API

```bash
mkdir -p data/raw
kaggle competitions download -c airbnb-recruiting-new-user-bookings -p data/raw
find data/raw -maxdepth 1 -name "*.zip" -exec unzip -q -o {} -d data/raw \;
```

Mantenha arquivos grandes fora do Git quando necessario e baixe-os novamente no Colab ou no ambiente local.
