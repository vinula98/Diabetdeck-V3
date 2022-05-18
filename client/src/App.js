import React, { useState, useEffect } from "react";
import "./Chatbot.css";
import DiabetesDay from "./components/images/DiabetesDay.jpg";
import DiabetdeckLogo from "./components/images/DiabetdeckLogo.png";
import DiabetesFacts from "./components/images/DiabetesFacts.png";
import Chatbot from "./components/images/Chatbot.png";
import User from "./components/images/User.png";
import Send from "./components/images/Send.png";
import Types from "./components/images/Types.jpg";
import Symptoms from "./components/images/Symptoms.jpg";
import Managing from "./components/images/Managing.jpg";

function App(props) {
  const [symptomData, setSymptomData] = useState("");
  const [symptomResponse, setSymptomResponse] = useState("");
  const [typedData, setTypedData] = useState("");
  const [finalDataState, setfinalDataState] = useState([{id:1, type:"sentMessage", message:"Hi! I'm Diabetdeck. In order to assist you please enter 3 symptoms you have out of the common Type 1 Diabetes symptoms shown in the screen."}]);
  const [imageIndex, setImageIndex] = useState(0);

  console.log(finalDataState)
  const getSymptomResponse = async (e) => {
    e.preventDefault();
    setTypedData(symptomData)
    // console.log(setTypedData)
    setSymptomData("")
    // console.log(setSymptomData)
    finalDataState.push( {id: "2", message: symptomData, type: "dataType"})
    console.log(symptomData)
    const response = await fetch("http://localhost:5000/predict", {
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

  const imagesSrc = [
    DiabetesDay,
    Symptoms,
    Managing
  ];

  // Slideshow Function
  const carousel = () => {
    setImageIndex((prev) => (prev < 2 ? prev + 1 : 0));
  };

  useEffect(() => {
    const timer = setInterval(() => {
      carousel();
    }, 2000); // set the which timer you want
    return () => {
      clearInterval(timer);
    };
  }, []);

  // Funtion to refresh page
  function refreshPage(){
    window.location.reload();
}

  return (
    <div className="App">
      <nav className="navbar navbar-expand-sm main-navigation">
        <div class="container-fluid">
          <a class="navbar-brand" href="#"><img src={DiabetdeckLogo} /></a>
        </div>
      </nav>

      <div className="container-fluid">
        <div className="row content">
          <div className="col-md-4">
            <div className="symptoms-box">
              <h3>Here are the Type 1 Diabetes Symptoms</h3>
              <div className="symptoms-list">
                <p>1. Polyuria</p>
                <p>2. Increased Appetite</p>
                <p>3. Excessive Hunger</p>
                <p>4. Obesity</p>
                <p>5. Blurred And Distorted Vision</p>
                <p>6. Irregular Sugar Level</p>
                <p>7. Lethargy</p>
                <p>8. Restlessness</p>
                <p>9. Weight Loss</p>
                <p>10. Fatigue</p>
              </div>
            </div>

            <div className="diabetes-facts-box">
              <img className="diabetes-facts" src={DiabetesFacts} />
            </div>
          </div>

          <div className="col-md-5">
            <div className="main-chatbot">
              <div className="message-header">
                <div className="row">
                  <div className="col-sm-1">
                    <div className="message-header-image">
                      <img className="chatbot-image" src={Chatbot} />
                    </div>
                  </div>
                  <div className="col-sm-3">
                    <div className="active">
                      <h4>Diabetdeck</h4>
                      <h6>Online Now</h6>
                    </div>
                  </div>
                  <div className="col-sm-6">

                  </div>
                  <div className="col-sm-2">
                    <div className="header-icons">
                      <button className="refresh-button" type="submit" onClick={refreshPage}>
                        <i class="fa fa-refresh" aria-hidden="true"></i>
                      </button>
                    </div>
                  </div>
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
                          <img className="recieved-chatbot-image" src={Chatbot} />
                          {/* <i class="fa fa-android" aria-hidden="true"></i> */}
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
                          <img className="user-image" src={User} />
                          {/* <i class="fa fa-android" aria-hidden="true"></i> */}
                        </div>
                      </div>
                        )
                      })}
                      
                    </div>
                  </div>
                </div>
              </div>

              <div className="chatbot-footer">
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
                        <button className="send-button" type="submit">
                          <img className="send-button-image" src={Send} />
                          {/* <i className="fa fa-paper-plane"></i> */}
                        </button>
                      </span>
                    </div>
                  </div>
                </form>
              </div>
            </div>
          </div>

          <div className="col-md-3">
            <div className="slideshow-container">
              <div className="diabetes-news">
                <img className="diabetes-day" src={imagesSrc[imageIndex]} alt="" />
              </div>
            </div>
            
            <div className="diabetes-types-box">
              <img className="diabetes-types-image" src={Types} />
            </div>
          </div>
        </div>
      </div>
      <footer class="container-fluid">
        <p className="footer-text">Copyright © 2022 Diabetdeck. - All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;
