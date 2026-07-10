import pandas as pd
import geopandas as gpd
import folium

df = pd.read_csv("../data/cleaned_house_data.csv")

gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["long"], df["lat"]),
    crs="EPSG:4326"
)

m = folium.Map(
    location=[df["lat"].mean(), df["long"].mean()],
    zoom_start=10
)

for _, row in df.head(500).iterrows():

    folium.CircleMarker(
        location=[row["lat"], row["long"]],
        radius=3,
        popup=f"Price: ${row['price']}",
        color="blue",
        fill=True
    ).add_to(m)

m.save("../results/house_map.html")

print("Map created successfully!")