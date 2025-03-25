const express = require("express");

const app = express();
const port = 3000;

app.use(express.json());

app.get("/api/hello", (req, res) => {
  res.json({
    message: "Hello from Beam!",
  });
});

app.post("/api/data", (req, res) => {
  const data = req.body;
  res.json({
    message: "Data received successfully",
    data: data,
  });
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
