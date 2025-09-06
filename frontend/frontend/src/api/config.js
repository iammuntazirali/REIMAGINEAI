// import axios from 'axios';

// const API_BASE = 'http://localhost:8000';

// const apiClient = axios.create({
//   baseURL: API_BASE,
//   headers: { 'Content-Type': 'application/json' },
// });

// apiClient.interceptors.response.use(
//   res => res,
//   err => {
//     if (!err.response) return Promise.reject(new Error('Network Error'));
//     if (err.response.status === 401) window.location = '/login';
//     return Promise.reject(err);
//   }
// );

// export default apiClient;
import axios from "axios";

const API_BASE = 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
});

apiClient.interceptors.response.use(
  (res) => res,
  (err) => {
    if (!err.response) {
      return Promise.reject(new Error('Network Error'));
    }
    if (err.response.status === 401) {
      window.location = '/login';
    }
    return Promise.reject(err);
  }
);

export default apiClient;
