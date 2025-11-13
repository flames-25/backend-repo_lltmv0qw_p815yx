import os
from io import BytesIO
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field

# ---------- App Setup ----------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Models ----------
class SearchRequest(BaseModel):
    address: str = Field(..., description="Street address to search around")
    county: str = Field("Denton County, TX", description="County and state")
    radius_miles: float = Field(2.0, gt=0, description="Search radius in miles")
    single_family_only: bool = Field(True, description="Filter to single-family homes")


class PropertyRecord(BaseModel):
    parcel_id: Optional[str] = None
    address: Optional[str] = None
    owner: Optional[str] = None
    land_value: Optional[float] = None
    improvement_value: Optional[float] = None
    total_appraised_value: Optional[float] = None
    year_built: Optional[int] = None
    lot_size: Optional[float] = Field(None, description="Lot size (sqft or acres depending on source)")
    legal_description: Optional[str] = None
    property_class: Optional[str] = None
    land_use: Optional[str] = None
    longitude: Optional[float] = None
    latitude: Optional[float] = None


# ---------- Helpers ----------
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "flames-blue-dcad-app/1.0 (contact: support@flames.blue)"


def geocode_address(address: str) -> Dict[str, float]:
    params = {
        "q": address,
        "format": "json",
        "limit": 1,
        "addressdetails": 1,
    }
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=20)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Geocoding failed: {r.text[:200]}")
    results = r.json()
    if not results:
        raise HTTPException(status_code=404, detail="Address not found")
    item = results[0]
    return {"lat": float(item["lat"]), "lon": float(item["lon"])}


class DentonCADClient:
    """
    Queries Denton County Appraisal District parcels via ArcGIS REST.
    Note: Endpoint may change; override via env ARCGIS_PARCELS_URL if needed.
    """

    # Best-effort default; can be overridden via env
    DEFAULT_LAYER_URL = os.getenv(
        "ARCGIS_PARCELS_URL",
        # If this specific layer URL changes in the future, set ARCGIS_PARCELS_URL env
        "https://gis.dentoncounty.gov/arcgis/rest/services/DCAD_public/MapServer/0",
    )

    def __init__(self) -> None:
        self.layer_url = self.DEFAULT_LAYER_URL
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    @staticmethod
    def miles_to_meters(miles: float) -> float:
        return miles * 1609.34

    def query_nearby(self, lon: float, lat: float, radius_miles: float) -> List[Dict[str, Any]]:
        params = {
            "f": "json",
            "where": "1=1",
            "geometry": f"{lon},{lat}",
            "geometryType": "esriGeometryPoint",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "distance": self.miles_to_meters(radius_miles),
            "units": "esriSRUnit_Meter",
            "outFields": "*",
            "outSR": 4326,
            "returnGeometry": True,
        }
        r = self.session.get(f"{self.layer_url}/query", params=params, timeout=30)
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Denton CAD query failed: {r.text[:200]}")
        data = r.json()
        if "error" in data:
            raise HTTPException(status_code=502, detail=f"Denton CAD error: {data['error']}")
        features = data.get("features", [])
        return features

    @staticmethod
    def is_single_family(attrs: Dict[str, Any]) -> bool:
        candidates = [
            str(attrs.get("LAND_USE", "")),
            str(attrs.get("LandUse", "")),
            str(attrs.get("PROPERTY_CLASS", "")),
            str(attrs.get("PropertyClass", "")),
            str(attrs.get("PROP_CLASS", "")),
            str(attrs.get("PropClass", "")),
            str(attrs.get("PROPERTY_TYPE", "")),
        ]
        text = " ".join(candidates).upper()
        return "SINGLE" in text or "SF" in text

    @staticmethod
    def norm_float(val: Any) -> Optional[float]:
        try:
            if val is None:
                return None
            return float(val)
        except Exception:
            return None

    @staticmethod
    def norm_int(val: Any) -> Optional[int]:
        try:
            if val is None:
                return None
            return int(float(val))
        except Exception:
            return None

    def normalize(self, feature: Dict[str, Any]) -> PropertyRecord:
        attrs = feature.get("attributes", {})
        geom = feature.get("geometry", {}) or {}
        x = geom.get("x")
        y = geom.get("y")

        # Try multiple common field names
        parcel_id = (
            attrs.get("PARCEL_ID")
            or attrs.get("ParcelID")
            or attrs.get("ACCOUNT")
            or attrs.get("Account")
            or attrs.get("OBJECTID")
        )
        address = (
            attrs.get("SITUS_ADDR")
            or attrs.get("SitusAddress")
            or attrs.get("SITUS")
            or attrs.get("Address")
        )
        owner = attrs.get("OWNER") or attrs.get("OwnerName") or attrs.get("OWNER_NAME")
        land_value = self.norm_float(
            attrs.get("LAND_VALUE") or attrs.get("LandValue") or attrs.get("LANDVAL")
        )
        impr_value = self.norm_float(
            attrs.get("IMPR_VALUE") or attrs.get("ImprovementValue") or attrs.get("IMPRVAL")
        )
        total_val = self.norm_float(
            attrs.get("TOTAL_VALUE")
            or attrs.get("TotalValue")
            or attrs.get("MKT_VAL")
            or attrs.get("APPR_VALUE")
            or attrs.get("ApprValue")
        )
        year_built = self.norm_int(attrs.get("YEAR_BUILT") or attrs.get("YearBuilt"))
        lot_size = self.norm_float(
            attrs.get("LOT_SIZE") or attrs.get("LotSize") or attrs.get("ACRES") or attrs.get("Acres")
        )
        legal_description = (
            attrs.get("LEGAL_DESC") or attrs.get("LegalDesc") or attrs.get("LEGAL_DESCRIPTION")
        )
        prop_class = attrs.get("PROPERTY_CLASS") or attrs.get("PropClass") or attrs.get("PROP_CLASS")
        land_use = attrs.get("LAND_USE") or attrs.get("LandUse")

        return PropertyRecord(
            parcel_id=str(parcel_id) if parcel_id is not None else None,
            address=str(address) if address is not None else None,
            owner=str(owner) if owner is not None else None,
            land_value=land_value,
            improvement_value=impr_value,
            total_appraised_value=total_val,
            year_built=year_built,
            lot_size=lot_size,
            legal_description=str(legal_description) if legal_description is not None else None,
            property_class=str(prop_class) if prop_class is not None else None,
            land_use=str(land_use) if land_use is not None else None,
            longitude=float(x) if isinstance(x, (int, float)) else None,
            latitude=float(y) if isinstance(y, (int, float)) else None,
        )


def to_excel(records: List[PropertyRecord]) -> bytes:
    import pandas as pd

    df = pd.DataFrame([r.model_dump() for r in records])
    # Order columns nicely
    preferred = [
        "parcel_id",
        "address",
        "owner",
        "total_appraised_value",
        "land_value",
        "improvement_value",
        "year_built",
        "lot_size",
        "legal_description",
        "property_class",
        "land_use",
        "latitude",
        "longitude",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Properties")
    buf.seek(0)
    return buf.read()


# ---------- Routes ----------
@app.get("/")
def read_root():
    return {"message": "Property search backend is running"}


@app.post("/api/properties/search", response_model=List[PropertyRecord])
def search_properties(req: SearchRequest):
    # Geocode
    coords = geocode_address(f"{req.address}, {req.county}")

    # Query CAD
    client = DentonCADClient()
    features = client.query_nearby(coords["lon"], coords["lat"], req.radius_miles)

    # Normalize
    records = [client.normalize(f) for f in features]

    # Filter single-family if requested
    if req.single_family_only:
        records = [r for r, f in zip(records, features) if client.is_single_family(f.get("attributes", {}))]

    return records


@app.post("/api/properties/export")
def export_properties(req: SearchRequest):
    # Reuse search logic
    results = search_properties(req)

    if not results:
        raise HTTPException(status_code=404, detail="No properties found for the specified criteria")

    xlsx = to_excel(results)

    filename = "denton_properties.xlsx"
    headers = {
        "Content-Disposition": f"attachment; filename={filename}",
        "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    return StreamingResponse(BytesIO(xlsx), headers=headers, media_type=headers["Content-Type"]) 


@app.get("/ui", response_class=HTMLResponse)
def simple_ui():
    # Lightweight UI served from backend as a fallback while frontend is unavailable
    html = """
    <!doctype html>
    <html>
    <head>
      <meta charset='utf-8'/>
      <meta name='viewport' content='width=device-width, initial-scale=1'/>
      <title>Denton County Property Finder</title>
      <style>
        body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:linear-gradient(135deg,#e0f2fe,#e0e7ff);margin:0;padding:0}
        .container{max-width:1100px;margin:0 auto;padding:24px}
        .card{background:#fff;border-radius:12px;box-shadow:0 1px 6px rgba(0,0,0,.08);padding:16px}
        .row{display:grid;grid-template-columns:1fr 160px auto;gap:12px;align-items:end}
        label{display:block;font-size:14px;font-weight:600;color:#374151}
        input[type=text],input[type=number]{margin-top:6px;width:100%;border:1px solid #d1d5db;border-radius:8px;padding:8px 10px}
        .btn{border:0;border-radius:8px;color:#fff;padding:8px 14px;font-weight:600;cursor:pointer}
        .btn-primary{background:#4f46e5}
        .btn-success{background:#059669}
        .muted{color:#6b7280}
        table{width:100%;border-collapse:collapse;font-size:14px}
        thead{background:#f9fafb}
        th,td{text-align:left;padding:10px 12px}
        tbody tr{border-top:1px solid #f3f4f6}
        .error{margin-top:12px;padding:10px;border-radius:8px;background:#fee2e2;color:#991b1b;border:1px solid #fecaca}
      </style>
    </head>
    <body>
      <div class='container'>
        <h1 style='font-size:28px;font-weight:800;color:#1f2937;margin:0 0 8px'>Denton County Property Finder</h1>
        <p class='muted' style='margin-top:0'>Enter an address and get nearby single-family homes with current appraisal values. Download as Excel in one click.</p>

        <div class='card'>
          <div class='row'>
            <div>
              <label>Address</label>
              <input id='address' type='text' placeholder='123 Main St, Denton, TX' />
            </div>
            <div>
              <label>Radius (miles)</label>
              <input id='radius' type='number' min='0.1' step='0.1' value='2'/>
            </div>
            <div style='display:flex;align-items:center;gap:8px'>
              <input id='sf' type='checkbox' checked />
              <label for='sf'>Single-family only</label>
            </div>
          </div>
          <div style='margin-top:12px;display:flex;gap:12px'>
            <button class='btn btn-primary' id='searchBtn'>Search</button>
            <button class='btn btn-success' id='exportBtn'>Download Excel</button>
          </div>
          <div id='error' class='error' style='display:none'></div>
        </div>

        <div class='card' style='margin-top:20px;overflow:auto'>
          <div style='padding-bottom:8px;border-bottom:1px solid #f3f4f6;display:flex;justify-content:space-between;align-items:center'>
            <h2 style='margin:0;font-weight:700;color:#1f2937'>Results <span id='count' class='muted'></span></h2>
          </div>
          <div style='overflow-x:auto'>
            <table>
              <thead>
                <tr>
                  <th>Parcel ID</th>
                  <th>Address</th>
                  <th>Owner</th>
                  <th>Total Value</th>
                  <th>Land Value</th>
                  <th>Impr. Value</th>
                  <th>Year Built</th>
                  <th>Lot Size</th>
                  <th>Class</th>
                  <th>Land Use</th>
                </tr>
              </thead>
              <tbody id='tbody'>
                <tr id='empty'>
                  <td colspan='10' class='muted' style='text-align:center;padding:32px 12px'>No results yet. Run a search above.</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <script>
        const API = ''
        const county = 'Denton County, TX'
        const $ = (id) => document.getElementById(id)
        const fmtUSD = (n) => (n != null ? Number(n).toLocaleString('en-US', { style: 'currency', currency: 'USD' }) : '-')

        function setError(msg){
          const el = $('error');
          if(!msg){ el.style.display='none'; el.textContent=''; return }
          el.style.display='block'; el.textContent = msg
        }
        function renderRows(rows){
          const tbody = $('tbody');
          tbody.innerHTML = ''
          if(!rows.length){
            const tr = document.createElement('tr');
            tr.innerHTML = `<td colspan="10" class="muted" style="text-align:center;padding:32px 12px">No results found.</td>`
            tbody.appendChild(tr)
            $('count').textContent = ''
            return
          }
          $('count').textContent = `(${rows.length})`
          for(const r of rows){
            const tr = document.createElement('tr')
            tr.innerHTML = `
              <td>${r.parcel_id ?? '-'}</td>
              <td>${r.address ?? '-'}</td>
              <td>${r.owner ?? '-'}</td>
              <td>${fmtUSD(r.total_appraised_value)}</td>
              <td>${fmtUSD(r.land_value)}</td>
              <td>${fmtUSD(r.improvement_value)}</td>
              <td>${r.year_built ?? '-'}</td>
              <td>${r.lot_size ?? '-'}</td>
              <td>${r.property_class ?? '-'}</td>
              <td>${r.land_use ?? '-'}</td>
            `
            tbody.appendChild(tr)
          }
        }

        $('searchBtn').addEventListener('click', async (e) => {
          e.preventDefault()
          setError('')
          const address = $('address').value.trim()
          const radius = parseFloat($('radius').value)
          const sf = $('sf').checked
          if(!address){ setError('Please enter an address.'); return }
          try{
            const res = await fetch(`${API}/api/properties/search`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ address, county, radius_miles: radius, single_family_only: sf })
            })
            if(!res.ok){
              const data = await res.json().catch(()=>({}))
              throw new Error(data.detail || `Request failed (${res.status})`)
            }
            const data = await res.json()
            renderRows(data)
          }catch(err){
            setError(err.message || 'Something went wrong')
          }
        })

        $('exportBtn').addEventListener('click', async (e) => {
          e.preventDefault()
          setError('')
          const address = $('address').value.trim()
          const radius = parseFloat($('radius').value)
          const sf = $('sf').checked
          if(!address){ setError('Please enter an address before exporting.'); return }
          try{
            const res = await fetch(`${API}/api/properties/export`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ address, county, radius_miles: radius, single_family_only: sf })
            })
            if(!res.ok){
              const data = await res.json().catch(()=>({}))
              throw new Error(data.detail || `Export failed (${res.status})`)
            }
            const blob = await res.blob()
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = 'denton_properties.xlsx'
            document.body.appendChild(a)
            a.click()
            a.remove()
            URL.revokeObjectURL(url)
          }catch(err){
            setError(err.message || 'Failed to download file')
          }
        })
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html, status_code=200)


@app.get("/test")
def test_database():
    """Simple health endpoint"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Used",
    }
    return response


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
