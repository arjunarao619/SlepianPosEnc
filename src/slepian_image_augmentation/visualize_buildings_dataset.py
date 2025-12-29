python -c "
import pandas as pd
df = pd.read_parquet('/scratch/local/arra4944_images/openbuildings/mexicocity_grid.parquet')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(df[['lon', 'lat', 'building_count', 'split']].head(10))
print(f'Lat range: [{df.lat.min():.4f}, {df.lat.max():.4f}]')
print(f'Lon range: [{df.lon.min():.4f}, {df.lon.max():.4f}]')
"