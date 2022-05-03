import React, { useState, useEffect } from "react";

function App(props) {
  const [symptomData, setSymptomData] = useState("");
  const [symptomResponse, setSymptomResponse] = useState("");

  const getSymptomResponse = async (e) => {
    e.preventDefault();
    const response = await fetch("http://localhost:5000/members", {
      method:'POST',
      headers:{'Accept': 'application/json', 'Content-Type': 'application/json'},
      body: JSON.stringify({
        symptomData
      })
    })

    const json = await response.json();
    setSymptomResponse(json.data);
  }

    //     fetch("/members")
    //     .then((res) =>res.json()
    //     .then((data) => {
    //             // Setting a data from api
    //             setdata({
    //               symptom_res: data.symptoms
    //             });
    //         })
    //     );
    // console.log(symptomData.symptom_res);

  return (
    <div className="App">
      <div className="container">
        <div>
          Response : {symptomResponse}
            <form onSubmit={(e) => getSymptomResponse(e)}>
                <input 
                  type='text' 
                  onChange={(e) => setSymptomData(e.target.value)} 
                  value={symptomData} 
                  className='form-control'
                />
                  <button type='submit'>submit</button>
            </form>
        </div> 
      </div>
    </div>
  );
}

export default App;
