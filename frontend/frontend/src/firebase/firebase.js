import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyCt0Dy0HdRf6ovWR01LscVYc74fkhEHk4o",
  authDomain: "fir-p-3f6ba.firebaseapp.com",
  projectId: "fir-p-3f6ba",
  storageBucket: "fir-p-3f6ba.firebasestorage.app",
  messagingSenderId: "149226079329",
  appId: "1:149226079329:web:e45588216533af75ecd620",
  measurementId: "G-MMNF0J18B1"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app)



export { app, auth };
