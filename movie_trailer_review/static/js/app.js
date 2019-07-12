function buildPlot() {
  /* data route */
var url = "/api/reviews";
d3.json(url).then(function(response) {

  console.log(response);

  var data = [response];

  var layout = {
    title: "Pet Pals",
    xaxis: {
      title: "Pet Type"
    },
    yaxis: {
      title: "Number of Pals"
    }
  };

  Plotly.newPlot("plot", data, layout);
});
}

buildPlot();
