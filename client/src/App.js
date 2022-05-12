import React, { useState, useEffect } from "react";
import "./Chatbot.css";

function App(props) {
  const [symptomData, setSymptomData] = useState("");
  const [symptomResponse, setSymptomResponse] = useState("");
  const [typedData, setTypedData] = useState("");
  const [finalDataState, setfinalDataState] = useState([{id:1, type:"sentMessage", message:"Hi! I'm Diabetdeck. In order to assist you please enter 3 symptoms you have out of the common Type 1 Diabetes symptoms shown in the screen."}]);

  console.log(finalDataState)
  const getSymptomResponse = async (e) => {
    e.preventDefault();
    setTypedData(symptomData)
    // console.log(setTypedData)
    setSymptomData("")
    // console.log(setSymptomData)
    finalDataState.push( {id: "2", message: symptomData, type: "dataType"})
    console.log(symptomData)
    const response = await fetch("http://localhost:5000/members", {
      method:'POST',
      headers:{'Accept': 'application/json', 'Content-Type': 'application/json'},
      body: JSON.stringify({
        symptomData
      })
    })

    const json = await response.json();
    setSymptomResponse(json.data);

    setfinalDataState([...finalDataState, {id: "2", message: json.data, type: "sentMessage"}])
  }

  return (
    <div className="App">
      <div className="container">
        <div className="message-header">
          <div className="message-header-image">
            {/* <img src=""/> */}
            <i class="fa fa-android" aria-hidden="true"></i>
          </div>
          <div className="active">
            <h4>Diabetdeck</h4>
            <h6>Online Now</h6>
          </div>
          <div className="header-icons">
            <i class="fa fa-refresh" aria-hidden="true"></i>
          </div>
        </div>

        <div className="chat-page">
          <div className="message-inbox">
            <div className="chats">
              <div className="message-page">
                {finalDataState.map ((data, index) => {
                  return (data.type === "sentMessage" ?
                  <div className="recieved-chat">
                  <div className="recieved-chat-image">
                    <i class="fa fa-android" aria-hidden="true"></i>
                  </div>
                  <div className="bot-message">
                    <div className="bot-message-inbox">
                      <p>{data.message}</p>
                    </div>
                  </div>
                </div>

                :
                
                <div className="user-chat">
                  <div className="user-message">
                    <p>{data.message}</p>
                  </div>
                  <div className="user-chat-image">
                    <i class="fa fa-android" aria-hidden="true"></i>
                  </div>
                </div>
                  )
                })}
                

                {/* {typedData && 
                
                } */}

                {/* {symptomResponse &&
                <div className="recieved-chat">
                  <div className="recieved-chat-image">
                    <i class="fa fa-android" aria-hidden="true"></i>
                  </div>
                  <div className="bot-message">
                    <div className="bot-message-inbox">
                      <p>{symptomResponse}</p>
                    </div>
                  </div>
                </div>
                } */}
                
              </div>
            </div>
          </div>
        </div>

        {/* <div className="chatbot-footer">
          <div className="input-group">
            
          </div>
          <input 
            type='text' 
            onChange={(e) => setSymptomData(e.target.value)} 
            value={symptomData} 
            className='form-control'
            placeholder="Type here.."
          />
        </div> */}

        <div className="chatbot-footer">
          {/* <div className="footer-icons">

          </div> */}
          <form onSubmit={(e) => getSymptomResponse(e)}>
            <div className="input-group">
              <input 
                type='text' 
                onChange={(e) => setSymptomData(e.target.value)} 
                value={symptomData} 
                className='form-control'
                placeholder="Type here.."
              />
              <div className="input-group-append">
                <span className="input-group-text">
                  <button type="submit"><i className="fa fa-paper-plane"></i></button>
                </span>
              </div>
            </div>
          </form>
        </div>
        
        {/* <div>
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
        </div>  */}
      </div>
    </div>
  );
}

export default App;
