import axios from "axios";

const api = axios.create({
  baseURL: "/api/v1",
  headers: { "Content-Type": "application/json" },
});

/* Attach JWT token from localStorage if present */
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("wm_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

/* ── Types ── */
export interface EmbedRequest {
  image_id: string;
  payload_hex?: string;
  sign?: boolean;
}

export interface EmbedResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface DetectRequest {
  image_id: string;
  verify_signature?: boolean;
}

export interface DetectResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface EmbedResult {
  watermarked_image_path: string;
  payload_hex: string;
  signature_hex: string | null;
  psnr_db: number | null;
}

export interface DetectResult {
  payload: string;
  confidence: number;
  attack_type: string;
  tamper_mask: string | null;
  latent_layer_intact: boolean;
  pixel_layer_intact: boolean;
  forgery_detected: boolean;
  bit_error_rate: number | null;
  ecdsa_valid: boolean | null;
}

export interface JobStatus {
  id: string;
  job_type: string;
  status: string;
  created_at: string;
  finished_at: string | null;
  error_message: string | null;
  result: EmbedResult | DetectResult | null;
}

export interface ImageUploadResponse {
  id: string;
  filename: string;
  filepath: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
}

/* ── API Methods ── */

export const authApi = {
  register: (email: string, password: string, full_name?: string) =>
    api.post("/auth/register", { email, password, full_name }),

  login: async (email: string, password: string): Promise<LoginResponse> => {
    const res = await api.post<LoginResponse>("/auth/login", {
      email,
      password,
    });
    localStorage.setItem("wm_token", res.data.access_token);
    return res.data;
  },

  logout: () => localStorage.removeItem("wm_token"),

  isLoggedIn: () => !!localStorage.getItem("wm_token"),
};

export const imageApi = {
  upload: async (file: File): Promise<ImageUploadResponse> => {
    const form = new FormData();
    form.append("file", file);
    const res = await api.post<ImageUploadResponse>("/images/upload", form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return res.data;
  },
};

export const watermarkApi = {
  embed: (req: EmbedRequest) => api.post<EmbedResponse>("/embed", req),
  detect: (req: DetectRequest) => api.post<DetectResponse>("/detect", req),
  jobStatus: (jobId: string) => api.get<JobStatus>(`/job/${jobId}`),
};

export default api;
