# -*- coding: utf-8 -*-

import os

import boto3
import netCDF4
import pandas as pd


def get_s3_bucket(name='envdataocean'):
    session = boto3.Session(region_name='cn-north-1',
                            aws_access_key_id='AKIAOJF3LJX3NSVU5VRQ',
                            aws_secret_access_key='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')
    bucket = session.resource('s3').Bucket(name)
    return bucket


def download_era5(i, i_lon, i_lat):
    s3_bucket = get_s3_bucket()
    s3_prefix = 'nfsdata/publicdata/ERA5/F.{}/{}/{}/'.format(i, i_lon, i_lat)
    ncs = sorted([os.path.basename(obj.key) for obj in s3_bucket.objects.filter(Prefix=s3_prefix)
                  if 'nc' in obj.key])
    [s3_bucket.download_file(s3_prefix + nc, nc) for nc in ncs]
    return ncs


def parse_nc_file(fn):
    nc = netCDF4.Dataset(fn)
    index = None
    data = {}
    for key in nc.variables.keys():
        value = nc.variables[key]
        if key == 'time':
            index = pd.date_range(netCDF4.num2date(value[0], units=value.units),
                                  freq='H', periods=len(value), name='timestamp')
        else:
            data[key] = value[:]
    nc.close()
    return pd.DataFrame(data, index=index)


def read_era5(lon, lat):
    idx_lon = int((lon + 180) / 0.25 + 1)
    idx_lat = int((lat + 90) / 0.25 + 1)
    file_name = 'ERA5.{}.{}.xlsx'.format(idx_lon, idx_lat)
    if os.path.exists(file_name):
        return pd.read_excel(file_name, index_col=0, parse_dates=True)

    ncs = download_era5(0, idx_lon, idx_lat)
    data_dfs = [parse_nc_file(nc) for nc in ncs]
    uv = pd.concat(data_dfs)
    [os.remove(nc) for nc in ncs]

    ncs = download_era5(1, idx_lon, idx_lat)
    data_dfs = [parse_nc_file(nc) for nc in ncs]
    pt = pd.concat(data_dfs)
    [os.remove(nc) for nc in ncs]

    data = pd.concat([uv, pt], axis=1)
    data["speed"] = (data["u100m"] ** 2 + data["v100m"] ** 2) ** 0.5
    # data.to_excel(file_name)
    return data


if __name__ == '__main__':
    read_era5(86.94022, 47.55122)
